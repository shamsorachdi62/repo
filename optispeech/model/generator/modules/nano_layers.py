from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from optispeech.model.generator.modules.pqmf import PQMF


def same_padding(kernel_size: int, dilation: int = 1) -> int:
    return dilation * (kernel_size - 1) // 2


def _normalize_padding_mask(mask: Tensor | None, batch_size: int, sequence_length: int, device: torch.device) -> Tensor | None:
    if mask is None:
        return None
    if mask.ndim != 2 or mask.shape != (batch_size, sequence_length):
        raise ValueError(
            f"padding_mask must have shape [{batch_size}, {sequence_length}], got {tuple(mask.shape)}"
        )
    return mask.to(device=device, dtype=torch.bool)


def _lengths_from_padding_mask(mask: Tensor | None) -> Tensor | None:
    if mask is None:
        return None
    lengths = mask.to(torch.int64).sum(dim=1)
    if torch.any(lengths <= 0):
        raise ValueError("padding_mask must leave at least one unmasked timestep in every batch item")
    return lengths.cpu()


def _mask_batch_time(x: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return x
    return x * mask.to(dtype=x.dtype).unsqueeze(-1)


def _mask_batch_channels_time(x: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return x
    return x * mask.to(dtype=x.dtype).unsqueeze(1)


def _upsample_padding_mask(mask: Tensor | None, factor: int, target_length: int) -> Tensor | None:
    if mask is None:
        return None
    upsampled = mask.repeat_interleave(factor, dim=1)
    if upsampled.shape[1] > target_length:
        return upsampled[:, :target_length]
    if upsampled.shape[1] < target_length:
        pad = upsampled.new_zeros(upsampled.shape[0], target_length - upsampled.shape[1])
        return torch.cat([upsampled, pad], dim=1)
    return upsampled


def _run_packed_recurrent(
    recurrent: nn.LSTM | nn.GRU,
    x: Tensor,
    state,
    padding_mask: Tensor | None,
) -> tuple[Tensor, Tensor | tuple[Tensor, Tensor]]:
    lengths = _lengths_from_padding_mask(padding_mask)
    if lengths is None:
        return recurrent(x, state)
    if bool(torch.all(lengths == x.shape[1])):
        return recurrent(x, state)

    packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
    packed_out, next_state = recurrent(packed, state)
    output, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.shape[1])
    return output, next_state


class GatedTemporalConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, groups: int = 32):
        super().__init__()
        self.proj = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=kernel_size,
            padding=same_padding(kernel_size),
            groups=groups,
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.glu(self.proj(x), dim=1)


class AcousticBlock(nn.Module):
    """Parallel ConvGLU + LSTM block used by the acoustic encoder and decoder."""

    def __init__(
        self,
        channels: int = 256,
        hidden_size: int = 128,
        bidirectional: bool = False,
        groups: int = 32,
    ):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.ones(channels))
        self.conv_stack = nn.ModuleList([GatedTemporalConv(channels, groups=groups) for _ in range(3)])
        self.recurrent = nn.LSTM(
            channels,
            hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
        )
        recurrent_channels = hidden_size * (2 if bidirectional else 1)
        if channels % recurrent_channels != 0:
            raise ValueError(f"channels={channels} must be divisible by recurrent_channels={recurrent_channels}")
        self.recurrent_tile = channels // recurrent_channels

    def forward(
        self,
        x: Tensor,
        state: tuple[Tensor, Tensor] | None = None,
        padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        padding_mask = _normalize_padding_mask(padding_mask, x.shape[0], x.shape[1], x.device)
        x = _mask_batch_time(x, padding_mask)

        y = x.transpose(1, 2)
        for conv in self.conv_stack:
            y = conv(y)
        conv_out = _mask_batch_time(y.transpose(1, 2), padding_mask)

        rnn_out, next_state = _run_packed_recurrent(self.recurrent, x, state, padding_mask)
        rnn_out = _mask_batch_time(rnn_out, padding_mask)
        if self.recurrent_tile != 1:
            rnn_out = rnn_out.repeat(1, 1, self.recurrent_tile)

        return _mask_batch_time(conv_out + rnn_out + x * self.residual_scale, padding_mask), next_state


class PhoneEncoder(nn.Module):
    def __init__(self, dim: int = 256, max_position: int = 16):
        super().__init__()
        self.blocks = nn.ModuleList([AcousticBlock(dim, hidden_size=128, bidirectional=True) for _ in range(3)])

    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        padding_mask = _normalize_padding_mask(~padding_mask, x.shape[0], x.shape[1], x.device)
        x = _mask_batch_time(x, padding_mask)
        for block in self.blocks:
            x, _ = block(x, padding_mask=padding_mask)
        return x


class DurationPredictor(nn.Module):
    def __init__(self, dim: int = 256, clip_val: float=1e-8):
        super().__init__()
        self.block = AcousticBlock(dim, hidden_size=64, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(dim, 1), nn.ReLU())
        self.clip_val = clip_val

    def forward(self, encoded: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        padding_mask = _normalize_padding_mask(~padding_mask, encoded.shape[0], encoded.shape[1], encoded.device)
        features, _ = self.block(encoded, padding_mask=padding_mask)
        return _mask_batch_time(self.head(features), padding_mask).squeeze(-1)

    @torch.inference_mode()
    def infer(self, encoded, padding_mask, factor=1.0):
        log_durations = self(encoded, padding_mask)
        # linear domain
        durations = torch.exp(log_durations) - self.clip_val
        durations = torch.ceil(durations * factor)
        # avoid negative values
        durations = torch.clamp(durations.long(), min=0)
        durations = durations.masked_fill(padding_mask, 0)
        return durations.clamp_max(80)


class PitchPredictor(nn.Module):
    """Nano pitch predictor with the duration predictor's acoustic backbone."""

    def __init__(self, dim: int = 256, embed_kernel_size: int = 9, embed_dropout: float = 0.2):
        super().__init__()
        self.block = AcousticBlock(dim, hidden_size=64, bidirectional=True)
        self.head = nn.Linear(dim, 1)
        self.embed = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=dim,
                kernel_size=embed_kernel_size,
                padding=same_padding(embed_kernel_size),
            ),
            nn.Dropout(embed_dropout),
        )

    def predict(self, encoded: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        mask = None if padding_mask is None else ~padding_mask
        valid_mask = _normalize_padding_mask(mask, encoded.shape[0], encoded.shape[1], encoded.device)
        features, _ = self.block(encoded, padding_mask=valid_mask)
        return _mask_batch_time(self.head(features), valid_mask).squeeze(-1)

    def condition(self, encoded: Tensor, pitch: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        mask = None if padding_mask is None else ~padding_mask
        valid_mask = _normalize_padding_mask(mask, encoded.shape[0], encoded.shape[1], encoded.device)
        pitch_embedding = self.embed(pitch.unsqueeze(1)).transpose(1, 2)
        return _mask_batch_time(encoded + pitch_embedding, valid_mask)

    def forward(
        self,
        encoded: Tensor,
        padding_mask: Tensor | None,
        target: Tensor,
    ) -> tuple[Tensor, Tensor]:
        prediction = self.predict(encoded, padding_mask)
        return self.condition(encoded, target, padding_mask), prediction

    @torch.inference_mode()
    def infer(
        self,
        encoded: Tensor,
        padding_mask: Tensor | None,
        factor: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        prediction = self.predict(encoded, padding_mask) * factor
        return self.condition(encoded, prediction, padding_mask), prediction


class DurationRegulator(nn.Module):
    def __init__(self, dim: int = 256, max_position: int = 16):
        super().__init__()
        self.alignment_smoother = nn.Conv2d(1, 1, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.forward_position = nn.Embedding(max_position, dim)
        self.backward_position = nn.Embedding(max_position, dim)

    def alignment_from_durations(self, encoded: Tensor, durations: Tensor) -> tuple[Tensor, Tensor]:
        durations = durations.to(torch.long)
        ends = torch.cumsum(durations, dim=1)
        max_len = int(ends.max().item()) if ends.numel() else 0
        if max_len == 0:
            empty_frames = encoded.new_zeros((encoded.shape[0], 0, encoded.shape[-1]))
            empty_alignment = encoded.new_zeros((encoded.shape[0], 0, encoded.shape[1]))
            return empty_frames, empty_alignment

        frame_positions = torch.arange(max_len, device=encoded.device)
        cumulative_mask = (frame_positions.view(1, 1, max_len) < ends.unsqueeze(-1)).to(encoded.dtype)
        previous_mask = torch.cat(
            [torch.zeros_like(cumulative_mask[:, :1, :]), cumulative_mask[:, :-1, :]],
            dim=1,
        )
        alignment = (cumulative_mask * (1.0 - previous_mask)).transpose(1, 2)
        smoothed_alignment = self.alignment_smoother(alignment.unsqueeze(1)).squeeze(1)
        return torch.matmul(smoothed_alignment, encoded), alignment

    def add_frame_positions(self, frames: Tensor, alignment: Tensor) -> Tensor:
        forward_cumsum = torch.cumsum(alignment, dim=1)
        forward_prev = torch.cat([torch.zeros_like(forward_cumsum[:, :1, :]), forward_cumsum[:, :-1, :]], dim=1)
        forward_index = (forward_prev * alignment).sum(dim=-1).to(torch.long)

        reversed_alignment = torch.flip(alignment, dims=(1,))
        backward_cumsum = torch.cumsum(reversed_alignment, dim=1)
        backward_prev = torch.cat([torch.zeros_like(backward_cumsum[:, :1, :]), backward_cumsum[:, :-1, :]], dim=1)
        backward_index = torch.flip(backward_prev * reversed_alignment, dims=(1,)).sum(dim=-1).to(torch.long)

        forward_index = forward_index.clamp_max(self.forward_position.num_embeddings - 1)
        backward_index = backward_index.clamp_max(self.backward_position.num_embeddings - 1)
        return frames + self.forward_position(forward_index) + self.backward_position(backward_index)

    def forward(self, encoded: Tensor, durations: Tensor) -> Tensor:
        frames, alignment = self.alignment_from_durations(encoded, durations)
        return self.add_frame_positions(frames, alignment)


class AcousticDecoder(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.blocks = nn.ModuleList(
            [AcousticBlock(dim, hidden_size=128, bidirectional=False) for _ in range(2)]
        )

    @staticmethod
    def zero_state(batch_size: int, device: torch.device | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        h1 = torch.zeros(1, batch_size, 128, device=device)
        c1 = torch.zeros(1, batch_size, 128, device=device)
        h2 = torch.zeros(1, batch_size, 128, device=device)
        c2 = torch.zeros(1, batch_size, 128, device=device)
        return h1, c1, h2, c2

    def forward(
        self,
        decoder_inputs: Tensor,
        padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        padding_mask = _normalize_padding_mask(
            ~padding_mask,
            decoder_inputs.shape[0],
            decoder_inputs.shape[1],
            decoder_inputs.device,
        )
        decoder_inputs = _mask_batch_time(decoder_inputs, padding_mask)
        x, (h1_new, c1_new) = self.blocks[0](decoder_inputs, padding_mask=padding_mask)
        x, (h2_new, c2_new) = self.blocks[1](x, padding_mask=padding_mask)
        return _mask_batch_time(x, padding_mask)


    @torch.inference_mode()
    def infer_streaming(
        self,
        decoder_inputs: Tensor,
        padding_mask: Tensor | None = None,
        h1: Tensor | None = None,
        c1: Tensor | None = None,
        h2: Tensor | None = None,
        c2: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        padding_mask = _normalize_padding_mask(
            ~padding_mask,
            decoder_inputs.shape[0],
            decoder_inputs.shape[1],
            decoder_inputs.device,
        )
        if h1 is None or c1 is None or h2 is None or c2 is None:
            h1, c1, h2, c2 = self.zero_state(decoder_inputs.shape[0], decoder_inputs.device)

        decoder_inputs = _mask_batch_time(decoder_inputs, padding_mask)
        x, (h1_new, c1_new) = self.blocks[0](decoder_inputs, (h1, c1), padding_mask=padding_mask)
        x, (h2_new, c2_new) = self.blocks[1](x, (h2, c2), padding_mask=padding_mask)
        return _mask_batch_time(x, padding_mask), h1_new, c1_new, h2_new, c2_new


class PaddedConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1, mode: str = "reflect"):
        super().__init__()
        self.pad = same_padding(kernel_size, dilation)
        self.mode = mode
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x: Tensor) -> Tensor:
        mode = "replicate" if self.mode == "edge" else self.mode
        return self.conv(F.pad(x, (self.pad, self.pad), mode=mode))

    def forward_streaming(self, x: Tensor, cache: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if self.pad == 0:
            return self.conv(x), x[..., :0]

        left = cache.to(device=x.device, dtype=x.dtype) if cache is not None and cache.shape[-1] == self.pad else x[..., :1].expand(-1, -1, self.pad)
        right = self._right_context(x)
        padded = torch.cat([left, x, right], dim=-1)
        history = torch.cat([left, x], dim=-1)
        return self.conv(padded), history[..., -self.pad :].detach()

    def _right_context(self, x: Tensor) -> Tensor:
        if self.mode == "edge" or x.shape[-1] == 1:
            return x[..., -1:].expand(-1, -1, self.pad)

        reflected = x[..., :-1].flip(-1)
        pieces = []
        while sum(piece.shape[-1] for piece in pieces) < self.pad:
            pieces.append(reflected if reflected.shape[-1] else x[..., -1:])
        return torch.cat(pieces, dim=-1)[..., : self.pad]


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(channels, 1))
        self.activation = nn.LeakyReLU(0.2)
        self.conv = PaddedConv1d(channels, channels, kernel_size=3, dilation=dilation, mode="reflect")

    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        padding_mask = _normalize_padding_mask(padding_mask, x.shape[0], x.shape[-1], x.device)
        x = _mask_batch_channels_time(x, padding_mask)
        y = self.conv(self.activation(x))
        return _mask_batch_channels_time(x + self.scale * y, padding_mask)

    def forward_streaming(self, x: Tensor, cache: Tensor | None = None) -> tuple[Tensor, Tensor]:
        y, next_cache = self.conv.forward_streaming(self.activation(x), cache)
        return x + self.scale * y, next_cache


class RationalBandHead(nn.Module):
    def __init__(self, channels: int = 32):
        super().__init__()
        self.proj = nn.Linear(channels, 1, bias=False)
        self.numerator_scale = nn.Parameter(torch.ones(1))
        self.denominator_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        y = self.proj(x)
        return self.numerator_scale * y / (self.denominator_scale + torch.abs(y))


class UpsampleStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, dilations: tuple[int, ...]):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=1,
            groups=4,
        )
        self.residual_blocks = nn.ModuleList([ResidualConvBlock(out_channels, dilation) for dilation in dilations])

    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        padding_mask = _normalize_padding_mask(padding_mask, x.shape[0], x.shape[-1], x.device)
        x = _mask_batch_channels_time(x, padding_mask)
        x = self.upsample(x)
        padding_mask = _upsample_padding_mask(padding_mask, self.upsample.stride[0], x.shape[-1])
        x = _mask_batch_channels_time(x, padding_mask)
        for block in self.residual_blocks:
            x = block(x, padding_mask)
        return x, padding_mask


@dataclass
class VocoderStreamState:
    gru0: Tensor | None = None
    gru1: Tensor | None = None
    conv: dict[str, Tensor] = field(default_factory=dict)
    deconv: dict[str, Tensor] = field(default_factory=dict)

    def detach(self) -> VocoderStreamState:
        return VocoderStreamState(
            gru0=None if self.gru0 is None else self.gru0.detach(),
            gru1=None if self.gru1 is None else self.gru1.detach(),
            conv={name: value.detach() for name, value in self.conv.items()},
            deconv={name: value.detach() for name, value in self.deconv.items()},
        )


class MultiBandVocoder(nn.Module):
    IS_F0_CONDITIONED: bool = False

    def __init__(
        self,
        input_channels: int = 100,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.prenet = nn.Sequential(PaddedConv1d(input_channels, 256, kernel_size=7, mode="edge"), nn.LeakyReLU(0.2))
        self.gru0 = nn.GRU(256, 128, batch_first=True)
        self.stage0 = UpsampleStage(256, 128, kernel_size=6, stride=3, padding=2, dilations=(1, 3, 9))
        self.stage1 = UpsampleStage(128, 64, kernel_size=10, stride=5, padding=3, dilations=(1, 3, 9))
        self.gru1 = nn.GRU(64, 64, batch_first=True)
        self.stage2 = UpsampleStage(64, 32, kernel_size=10, stride=5, padding=3, dilations=(1, 3, 9, 27, 81))
        self.final_feature_conv = PaddedConv1d(32, 32, kernel_size=7, mode="edge")
        self.band_heads = nn.ModuleList([RationalBandHead(32) for _ in range(4)])
        self.pqmf = PQMF(subbands=4, taps=62, cutoff_ratio=0.142, beta=9.0)

    def forward(self, cond: Tensor, f0: Tensor, padding_mask=None) -> Tensor:
        out, *_states= self._vocode(cond, padding_mask=padding_mask)
        return out

    def _vocode(
        self,
        mel: Tensor,
        state1: Tensor | None = None,
        state2: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        padding_mask = _normalize_padding_mask(~padding_mask, mel.shape[0], mel.shape[-1], mel.device)
        x = self.prenet(mel)
        x = _mask_batch_channels_time(x, padding_mask)

        gru0_state = None if state1 is None else state1.unsqueeze(0)
        gru0_out, state1_out = _run_packed_recurrent(self.gru0, x.transpose(1, 2), gru0_state, padding_mask)
        x = x + gru0_out.repeat(1, 1, 2).transpose(1, 2)
        x = _mask_batch_channels_time(x, padding_mask)

        x, padding_mask = self.stage0(x, padding_mask)
        x, padding_mask = self.stage1(F.leaky_relu(x, 0.2), padding_mask)

        gru1_state = None if state2 is None else state2.unsqueeze(0)
        gru1_out, state2_out = _run_packed_recurrent(self.gru1, x.transpose(1, 2), gru1_state, padding_mask)
        x = x + gru1_out.transpose(1, 2)
        x = _mask_batch_channels_time(x, padding_mask)

        x, padding_mask = self.stage2(F.leaky_relu(x, 0.2), padding_mask)
        x = F.leaky_relu(self.final_feature_conv(F.leaky_relu(x, 0.2)), 0.2)
        x = _mask_batch_channels_time(x, padding_mask)

        bands = [head(x.transpose(1, 2)).transpose(1, 2) for head in self.band_heads]
        bands = torch.cat(bands, dim=1)
        bands = _mask_batch_channels_time(bands, padding_mask)
        wave = self.pqmf.synthesis(bands)
        padding_mask = _upsample_padding_mask(padding_mask, self.pqmf.subbands, wave.shape[-1])
        wave = _mask_batch_channels_time(wave, padding_mask)
        return wave.squeeze(1), state1_out.squeeze(0), state2_out.squeeze(0)

    def initial_streaming_state(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> VocoderStreamState:
        kwargs = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return VocoderStreamState(
            gru0=torch.zeros(batch_size, 128, **kwargs),
            gru1=torch.zeros(batch_size, 64, **kwargs),
        )

    def forward_streaming(
        self,
        mel: Tensor,
        state: VocoderStreamState | None = None,
        emit_len: int | Tensor | None = None,
    ) -> tuple[Tensor, VocoderStreamState]:
        if emit_len is None:
            emit_len = mel.shape[-1]
        if isinstance(emit_len, Tensor):
            emit_len_value = int(emit_len.item())
        else:
            emit_len_value = int(emit_len)

        state = state or self.initial_streaming_state(mel.shape[0], mel.device, mel.dtype)
        if state.gru0 is None:
            state.gru0 = torch.zeros(mel.shape[0], 128, device=mel.device, dtype=mel.dtype)
        if state.gru1 is None:
            state.gru1 = torch.zeros(mel.shape[0], 64, device=mel.device, dtype=mel.dtype)

        x, state.conv["prenet"] = self.prenet[0].forward_streaming(mel, state.conv.get("prenet"))
        x = self.prenet[1](x)
        gru0_out, next_gru0 = self.gru0(x.transpose(1, 2), state.gru0.unsqueeze(0))
        x = x + gru0_out.repeat(1, 1, 2).transpose(1, 2)

        x = self._stream_stage("stage0", self.stage0, x, state)
        x = self._stream_stage("stage1", self.stage1, F.leaky_relu(x, 0.2), state)

        gru1_out, next_gru1 = self.gru1(x.transpose(1, 2), state.gru1.unsqueeze(0))
        x = x + gru1_out.transpose(1, 2)

        x = self._stream_stage("stage2", self.stage2, F.leaky_relu(x, 0.2), state)
        x, state.conv["final_feature"] = self.final_feature_conv.forward_streaming(
            F.leaky_relu(x, 0.2),
            state.conv.get("final_feature"),
        )
        x = F.leaky_relu(x, 0.2)

        bands = [head(x.transpose(1, 2)).transpose(1, 2) for head in self.band_heads]
        bands = torch.cat(bands, dim=1)
        wave, state.conv["pqmf_synthesis"] = self._stream_pqmf_synthesis(
            bands,
            state.conv.get("pqmf_synthesis"),
            emit_len_value * 75,
        )
        wave = wave[..., : emit_len_value * 300]

        state.gru0 = next_gru0.squeeze(0).detach()
        state.gru1 = next_gru1.squeeze(0).detach()
        return wave, state.detach()

    def _stream_stage(self, name: str, stage: UpsampleStage, x: Tensor, state: VocoderStreamState) -> Tensor:
        x = self._stream_deconv(f"{name}.upsample", stage.upsample, x, state)
        for index, block in enumerate(stage.residual_blocks):
            key = f"{name}.residual.{index}"
            x, state.conv[key] = block.forward_streaming(x, state.conv.get(key))
        return x

    @staticmethod
    def _deconv_context(module: nn.ConvTranspose1d) -> int:
        return max(1, (module.kernel_size[0] + module.stride[0] - 1) // module.stride[0])

    def _stream_deconv(
        self,
        name: str,
        module: nn.ConvTranspose1d,
        x: Tensor,
        state: VocoderStreamState,
    ) -> Tensor:
        cache = state.deconv.get(name)
        if cache is None:
            cache = x[..., :0]
        else:
            cache = cache.to(device=x.device, dtype=x.dtype)
        combined = torch.cat([cache, x], dim=-1)
        y = module(combined)[..., cache.shape[-1] * module.stride[0] :]
        state.deconv[name] = combined[..., -self._deconv_context(module) :].detach()
        return y

    def _stream_pqmf_synthesis(
        self,
        x: Tensor,
        cache: Tensor | None = None,
        commit_len: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        pad = 31
        commit_len = x.shape[-1] if commit_len is None else commit_len
        x = F.conv_transpose1d(x, self.pqmf.updown_filter * self.pqmf.subbands, stride=self.pqmf.subbands)
        left = (
            cache.to(device=x.device, dtype=x.dtype)
            if cache is not None and cache.shape[1] == x.shape[1] and cache.shape[-1] == pad
            else x.new_zeros(x.shape[0], x.shape[1], pad)
        )
        padded = torch.cat([left, x, x.new_zeros(x.shape[0], x.shape[1], pad)], dim=-1)
        sample_commit_len = commit_len * self.pqmf.subbands
        return F.conv1d(padded, self.pqmf.synthesis_filter), torch.cat([left, x[..., :sample_commit_len]], dim=-1)[..., -pad:].detach()


@dataclass(frozen=True)
class HamedIdiomaticModels:
    encoder: PhoneEncoder
    duration_predictor: DurationPredictor
    duration_regulator: DurationRegulator
    decoder: AcousticDecoder
    vocoder: MultiBandVocoder


def _copy_acoustic_block(target: AcousticBlock, source) -> None:
    with torch.no_grad():
        target.residual_scale.copy_(source.alpha)
        for target_conv, source_conv in zip(target.conv_stack, source.ffn):
            target_conv.proj.weight.copy_(source_conv.conv.weight)
            target_conv.proj.bias.copy_(source_conv.conv.bias)
        target.recurrent.load_state_dict(source.lstm.state_dict())


def copy_encoder_from_reference(target: PhoneEncoder, source) -> PhoneEncoder:
    with torch.no_grad():
        target.embedding.weight.copy_(source.embedding.weight)
        for target_block, source_block in zip(target.blocks, (source.enc_0, source.enc_1, source.enc_2)):
            _copy_acoustic_block(target_block, source_block)
    return target


def copy_duration_predictor_from_reference(target: DurationPredictor, source) -> DurationPredictor:
    with torch.no_grad():
        _copy_acoustic_block(target.block, source.duration_rnn)
        target.head[0].weight.copy_(source.duration_post_layer[0].weight)
        target.head[0].bias.copy_(source.duration_post_layer[0].bias)
    return target


def copy_duration_regulator_from_reference(target: DurationRegulator, source) -> DurationRegulator:
    with torch.no_grad():
        target.alignment_smoother.weight.copy_(source.smoother.weight)
        target.forward_position.weight.copy_(source.fw_pos_embedding.weight)
        target.backward_position.weight.copy_(source.bw_pos_embedding.weight)
    return target


def copy_decoder_from_reference(target: AcousticDecoder, source) -> AcousticDecoder:
    with torch.no_grad():
        for target_block, source_block in zip(target.blocks, (source.dec_0, source.dec_1)):
            _copy_acoustic_block(target_block, source_block)
        target.to_mel.weight.copy_(source.mel_proj.weight)
        target.to_mel.bias.copy_(source.mel_proj.bias)
    return target


def _copy_residual(target: ResidualConvBlock, source) -> None:
    target.scale.data.copy_(source.alpha)
    target.conv.conv.weight.data.copy_(source.block[1].conv.weight)
    target.conv.conv.bias.data.copy_(source.block[1].conv.bias)


def _copy_stage(target: UpsampleStage, source_deconv: nn.ConvTranspose1d, source_blocks: tuple) -> None:
    target.upsample.load_state_dict(source_deconv.state_dict())
    for target_block, source_block in zip(target.residual_blocks, source_blocks):
        _copy_residual(target_block, source_block)


def copy_vocoder_from_reference(target: MultiBandVocoder, source) -> MultiBandVocoder:
    with torch.no_grad():
        target.prenet[0].conv.load_state_dict(source.prenet[0].conv.state_dict())
        target.gru0.load_state_dict(source.rnn0.state_dict())
        _copy_stage(target.stage0, source.ups_0, (source.ups_1, source.ups_2, source.ups_3))
        _copy_stage(target.stage1, source.ups_5, (source.ups_6, source.ups_7, source.ups_8))
        target.gru1.load_state_dict(source.rnn.state_dict())
        _copy_stage(target.stage2, source.ups_10, (source.ups_11, source.ups_12, source.ups_13, source.ups_14, source.ups_15))
        target.final_feature_conv.conv.load_state_dict(source.ups_18.conv.state_dict())
        for target_head, source_head in zip(target.band_heads, source.band_linear_layer):
            target_head.proj.weight.copy_(source_head.linear.weight)
            target_head.numerator_scale.copy_(source_head.a)
            target_head.denominator_scale.copy_(source_head.b)
    return target


def build_from_reference(reference_models) -> HamedIdiomaticModels:
    return HamedIdiomaticModels(
        encoder=copy_encoder_from_reference(PhoneEncoder(), reference_models.encoder),
        duration_predictor=copy_duration_predictor_from_reference(DurationPredictor(), reference_models.encoder),
        duration_regulator=copy_duration_regulator_from_reference(DurationRegulator(), reference_models.encoder),
        decoder=copy_decoder_from_reference(AcousticDecoder(), reference_models.decoder),
        vocoder=copy_vocoder_from_reference(MultiBandVocoder(), reference_models.vocoder),
    )


def _assert_close(name: str, actual: Tensor, expected: Tensor, atol: float = 1e-6) -> None:
    diff = (actual - expected).abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    if not torch.allclose(actual, expected, atol=atol, rtol=0.0):
        raise AssertionError(f"{name} mismatch: max_abs={max_abs}")
    print(f"{name}: ok max_abs={max_abs:.3g}")


def _report_difference(name: str, actual: Tensor, expected: Tensor) -> None:
    diff = (actual - expected).abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    mean_abs = float(diff.mean()) if diff.numel() else 0.0
    print(f"{name}: max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}")


def validate_against_reference() -> None:
    from hamed_pytorch import (
        HamedDecoder,
        HamedEncoder,
        HamedModels,
        HamedVocoder,
        import_hamed_decoder_weights,
        import_hamed_encoder_weights,
        import_hamed_vocoder_weights,
    )
    torch.manual_seed(1234)
    reference = HamedModels(
        encoder=import_hamed_encoder_weights(HamedEncoder(), "hamd/hamed_encoder.onnx").eval(),
        decoder=import_hamed_decoder_weights(HamedDecoder(), "hamd/hamed_decoder.onnx").eval(),
        vocoder=import_hamed_vocoder_weights(HamedVocoder(), "hamd/hamed_vocoder.onnx").eval(),
    )
    idiomatic = build_from_reference(reference)
    idiomatic.encoder.eval()
    idiomatic.duration_predictor.eval()
    idiomatic.duration_regulator.eval()
    idiomatic.decoder.eval()
    idiomatic.vocoder.eval()

    with torch.no_grad():
        phone_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.int32)
        speed = torch.ones(1, 8)
        ref_decoder_inputs, ref_durations = reference.encoder(phone_ids, speed)
        encoded = idiomatic.encoder(phone_ids)
        new_durations = idiomatic.duration_predictor(encoded, speed)
        new_decoder_inputs = idiomatic.duration_regulator(encoded, new_durations)
        if not torch.equal(new_durations, ref_durations):
            raise AssertionError(f"duration mismatch: {new_durations} != {ref_durations}")
        _assert_close("encoder.decoder_inputs", new_decoder_inputs, ref_decoder_inputs)

        decoder_inputs = torch.randn(1, 12, 256)
        states = tuple(torch.randn(1, 1, 128) for _ in range(4))
        for index, (actual, expected) in enumerate(zip(idiomatic.decoder(decoder_inputs, *states), reference.decoder(decoder_inputs, *states))):
            _assert_close(f"decoder.output[{index}]", actual, expected)

        mel = torch.randn(1, 80, 10)
        state1 = torch.randn(1, 128)
        state2 = torch.randn(1, 64)
        for index, (actual, expected) in enumerate(zip(idiomatic.vocoder(mel, state1, state2), reference.vocoder(mel, state1, state2))):
            if index == 0:
                _report_difference("vocoder.wave.generated_pqmf_vs_extracted", actual, expected)
            else:
                _assert_close(f"vocoder.output[{index}]", actual, expected)

        ref_stream_state = None
        new_stream_state = None
        for step, chunk in enumerate(mel.split([3, 2, 5], dim=-1)):
            ref_wave, ref_stream_state = reference.vocoder.forward_streaming(chunk, ref_stream_state)
            new_wave, new_stream_state = idiomatic.vocoder.forward_streaming(chunk, new_stream_state)
            _report_difference(f"vocoder.streaming.generated_pqmf_vs_extracted[{step}]", new_wave, ref_wave)


if __name__ == "__main__":
    validate_against_reference()
