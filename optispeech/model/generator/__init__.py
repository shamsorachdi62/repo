from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from optispeech.utils import denormalize, sequence_mask
from optispeech.utils.segments import get_segments, get_random_segments

from.modules.nano_layers import DurationRegulator
from .alignments import (
    AlignmentModule,
    average_by_duration,
    expand_by_duration,
    viterbi_decode,
)
from .loss import FastSpeech2Loss, ForwardSumLoss
from .nets_utils import make_pad_mask


class OptiSpeechGenerator(nn.Module):
    def __init__(
        self,
        dim: int,
        segment_size,
        text_embedding,
        encoder,
        duration_predictor,
        pitch_predictor,
        decoder,
        vocoder,
        loss_coeffs,
        feature_extractor,
        num_speakers,
        num_languages,
        data_statistics,
        pause_token_ids=(),
        **kwargs
    ):
        super().__init__()

        self.segment_size = segment_size
        self.loss_coeffs = loss_coeffs
        self.n_feats = feature_extractor.n_feats
        self.n_fft = feature_extractor.n_fft
        self.hop_length = feature_extractor.hop_length
        self.sample_rate = feature_extractor.sample_rate
        self.data_statistics = data_statistics
        self.num_speakers = num_speakers
        self.num_languages = num_languages
        self.register_buffer(
            "pause_token_ids",
            torch.as_tensor(pause_token_ids, dtype=torch.long),
            persistent=False,
        )

        self.text_embedding = text_embedding(dim=dim)
        self.encoder = encoder(dim=dim)
        self.duration_predictor = duration_predictor(dim=dim)
        self.text_align_proj = nn.Linear(dim, self.n_feats)
        self.alignment_module = AlignmentModule(adim=self.n_feats, odim=self.n_feats)
        self.pitch_predictor = pitch_predictor(dim=dim)
        self.feature_upsampler = DurationRegulator(dim)
        self.decoder = decoder(dim=dim)
        self.vocoder = vocoder(
            input_channels=dim // 2,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        self.dec_proj = nn.Linear(dim, dim // 2)
        if self.num_speakers > 1:
            self.sid_embed = torch.nn.Embedding(self.num_speakers, dim)
        if self.num_languages > 1:
            self.lid_embed = torch.nn.Embedding(self.num_languages, dim)
        self.loss_criterion = FastSpeech2Loss()
        self.forwardsum_loss = ForwardSumLoss()

    def forward(self, x, x_lengths, mel, mel_lengths, pitches, sids, lids):
        """
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            pitches (torch.Tensor): phoneme-level pitch values.
                shape: (batch_size, max_text_length)
            sids (torch.LongTensor): list of speaker IDs for each input sentence.
                shape: (batch_size,)
            lids (torch.LongTensor): list of language IDs for each input sentence.
                shape: (batch_size,)

        Returns:
            loss: (torch.Tensor): scaler representing total loss
            alignment_loss: (torch.Tensor): scaler representing alignment loss
            duration_loss: (torch.Tensor): scaler representing durations loss
            pitch_loss: (torch.Tensor): scaler representing pitch loss
        """
        f0_frame_level = pitches
        x_max_length = x_lengths.max()
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x_max_length), 1).type_as(x)

        mel_max_length = mel_lengths.max()
        mel_mask = torch.unsqueeze(sequence_mask(mel_lengths, mel_max_length), 1).type_as(x)

        input_padding_mask = ~x_mask.squeeze(1).bool().to(x.device)
        target_padding_mask = ~mel_mask.squeeze(1).bool().to(x.device)

        # text embedding
        x, __ = self.text_embedding(x)

        # Encoder
        x = self.encoder(x, input_padding_mask)

        # Speaker and language embedding
        if sids is not None:
            sid_emb = self.sid_embed(sids.view(-1))
            x = x + sid_emb.unsqueeze(1)
        if lids is not None:
            lid_embs = self.lid_embed(lids.view(-1))
            x = x + lid_embs.unsqueeze(1)

        # alignment
        h_masks = make_pad_mask(x_lengths)
        text_align_p = self.text_align_proj(x)
        log_p_attn = self.alignment_module(
            text=text_align_p,
            feats=mel.transpose(1, 2),
            text_lengths=x_lengths,
            feats_lengths=mel_lengths,
            x_masks=h_masks,
        )
        durations, bin_loss = viterbi_decode(log_p_attn, x_lengths, mel_lengths)
        forwardsum_loss = self.forwardsum_loss(log_p_attn, x_lengths, mel_lengths)
        align_loss = forwardsum_loss + bin_loss

        # Average pitch values based on durations
        pitches = average_by_duration(durations, pitches.unsqueeze(-1), x_lengths, mel_lengths)

        # token-level pitch predictor
        x, pitch_hat = self.pitch_predictor(x, input_padding_mask, pitches)

        # Duration predictor
        duration_hat = self.duration_predictor(x.detach(), input_padding_mask)

        # upsample to mel lengths
        y = self.feature_upsampler(x, durations)

        # Decoder
        y = self.decoder(y, target_padding_mask)
        y = self.dec_proj(y)

        # get random segments
        segment_size = min(self.segment_size, y.shape[-2])
        num_frames = mel_lengths - 4 # mel-centered
        segment, start_idx = get_random_segments(
            y.transpose(1, 2),
            num_frames.type_as(y),
            segment_size,
        )
        # F0
        if self.vocoder.IS_F0_CONDITIONED:
            f0_cond = get_segments(
                f0_frame_level.unsqueeze(1),
                start_idxs=start_idx,
                segment_size=segment_size
            )
            # Descale f0
            f0_cond = denormalize(f0_cond, self.data_statistics.pitch_mean, self.data_statistics.pitch_std)
        else:
            f0_cond = None
        # Generate wav
        wav_hat = self.vocoder(
            segment,
            f0=f0_cond,
            padding_mask=torch.zeros(segment.shape[0], segment.shape[-1]).long().to(segment.device),
        )

        # Losses
        loss_coeffs = self.loss_coeffs
        duration_loss, pitch_loss = self.loss_criterion(
            d_outs=duration_hat.unsqueeze(-1),
            p_outs=pitch_hat.unsqueeze(-1),
            ds=durations.unsqueeze(-1),
            ps=pitches.unsqueeze(-1),
            ilens=x_lengths,
        )
        loss = (
            (align_loss * loss_coeffs.lambda_align)
            + (duration_loss * loss_coeffs.lambda_duration)
            + (pitch_loss * loss_coeffs.lambda_pitch)
        )

        return {
            "wav_hat": wav_hat,
            "start_idx": start_idx,
            "segment_size": segment_size,
            "loss": loss,
            "align_loss": align_loss.detach().cpu(),
            "duration_loss": duration_loss.detach().cpu(),
            "pitch_loss": pitch_loss.detach().cpu(),
        }

    @torch.inference_mode()
    def _adjust_pause_durations(self, token_ids, durations, pause_factor, pause_mask=None):
        if pause_factor < 1.0:
            raise ValueError("pause_factor must be greater than or equal to 1.0")
        if pause_factor == 1.0 or self.pause_token_ids.numel() == 0:
            return durations

        if pause_mask is None:
            pause_mask = (token_ids.unsqueeze(-1) == self.pause_token_ids).any(dim=-1)
        else:
            pause_mask = pause_mask.bool()
        scaled_durations = torch.ceil(durations.to(torch.float32) * pause_factor).to(durations.dtype)
        return torch.where(pause_mask, scaled_durations, durations)

    @torch.inference_mode()
    def synthesise(
        self,
        x,
        x_lengths,
        sids=None,
        lids=None,
        d_factor=1.0,
        p_factor=1.0,
        pause_factor=2.0,
        pause_mask=None,
    ):
        """
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            sids (Optional[torch.LongTensor]): list of speaker IDs for each input sentence.
                shape: (batch_size,)
            lids (Optional[torch.LongTensor]): list of language IDs for each input sentence.
                shape: (batch_size,)
            d_factor (Optional[float]): scaler to control phoneme durations.
            p_factor (Optional[float]): scaler to control pitch.
            pause_factor (Optional[float]): duration multiplier for pause tokens.
            pause_mask (Optional[torch.BoolTensor]): tokens whose durations should use the pause multiplier.

        Returns:
            wav (torch.Tensor): generated waveform
                shape: (batch_size, T)
            durations: (torch.Tensor): predicted phoneme durations
                shape: (batch_size, max_text_length)
            pitch: (torch.Tensor): predicted pitch
                shape: (batch_size, max_text_length)
            rtf: (float): total Realtime Factor (inference_t/audio_t)
        """
        am_t0 = perf_counter()
        token_ids = x

        x_max_length = x_lengths.max()
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x_max_length), 1).to(x.dtype)
        x_mask = x_mask.to(x.device)
        input_padding_mask = ~x_mask.squeeze(1).bool().to(x.device)

        # text embedding
        x, __ = self.text_embedding(x)

        # Encoder
        x = self.encoder(x, input_padding_mask)

        # Set default speaker/language during inference when not specified
        if (self.num_speakers > 1) and sids is None:
            sids = torch.zeros(x.shape[0]).long().to(x.device)
        if (self.num_languages > 1) and lids is None:
            lids = torch.zeros(x.shape[0]).long().to(x.device)

        # Speaker and language embedding
        if sids is not None:
            sid_emb = self.sid_embed(sids.view(-1))
            x = x + sid_emb.unsqueeze(1)
        if lids is not None:
            lid_embs = self.lid_embed(lids.view(-1))
            x = x + lid_embs.unsqueeze(1)

        # pitch predictor
        x, pitch = self.pitch_predictor.infer(x, input_padding_mask, p_factor)

        # duration predictor
        durations = self.duration_predictor.infer(x, input_padding_mask, factor=d_factor)
        durations = self._adjust_pause_durations(token_ids, durations, pause_factor, pause_mask)

        y_lengths = durations.sum(dim=1)
        y_max_length = y_lengths.max()
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).type_as(x)
        target_padding_mask = ~y_mask.squeeze(1).bool()

        y = self.feature_upsampler(x, durations)

        # Decoder
        y = self.decoder(y, target_padding_mask)
        y = self.dec_proj(y).transpose(1, 2)
        am_infer = (perf_counter() - am_t0) * 1000

        v_t0 = perf_counter()
        # Generate wav
        if self.vocoder.IS_F0_CONDITIONED:
            frame_pitch, _ = expand_by_duration(pitch.unsqueeze(-1), durations)
            f0_cond = denormalize(frame_pitch.transpose(1, 2), self.data_statistics.pitch_mean, self.data_statistics.pitch_std)
        else:
            f0_cond = None
        wav = self.vocoder(
            y,
            f0=f0_cond,
            padding_mask=target_padding_mask
        )
        wav_lengths = y_lengths * self.hop_length
        v_infer = (perf_counter() - v_t0) * 1000

        wav_t = wav.shape[-1] / (self.sample_rate * 1e-3)
        am_rtf = am_infer / wav_t
        v_rtf = v_infer / wav_t
        rtf = am_rtf + v_rtf
        latency = am_infer + v_infer

        return {
            "wav": wav.detach().cpu(),
            "wav_lengths": wav_lengths.detach().cpu(),
            "durations": durations.detach().cpu(),
            "pitch": pitch.detach().cpu(),
            "am_rtf": am_rtf,
            "v_rtf": v_rtf,
            "rtf": rtf,
            "latency": latency,
        }
