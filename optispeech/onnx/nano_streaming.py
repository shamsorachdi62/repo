import torch
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat
if hasattr(torch.serialization, 'add_safe_globals'):
    try:
        import hydra._internal.target_policy
        torch.serialization.add_safe_globals([hydra._internal.target_policy._DeferredTarget])
    except Exception:
        pass

import argparse
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import torch.nn.functional as F
from torch import nn

from optispeech.model import OptiSpeech
from optispeech.utils import get_script_logger, sequence_mask


log = get_script_logger(__name__)

DEFAULT_TEXT = "مرحبا، هذا اختبار سريع لقياس زمن أول عينة صوتية."
DEFAULT_OPSET = 16
DEFAULT_RUNS_DIR = Path("logs/train/salim-nano/runs")
QUANTIZED_OP_TYPES = ["MatMul", "Gemm", "LSTM"]


def find_latest_checkpoint(runs_dir: Path) -> Path:
    checkpoints = [path for path in runs_dir.rglob("*.ckpt") if path.is_file()]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {runs_dir}")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def load_model_for_export(checkpoint: Path):
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint_data["state_dict"]
    legacy_synthesis = "generator.vocoder.subband_upsample.weight" in state_dict
    model = OptiSpeech.load_from_checkpoint(
        checkpoint,
        map_location="cpu",
        strict=not legacy_synthesis,
    )
    if legacy_synthesis:
        vocoder = model.generator.vocoder
        vocoder.subband_upsample = nn.ConvTranspose1d(4, 4, kernel_size=4, stride=4, bias=False)
        vocoder.synthesis_filter = nn.Conv1d(4, 1, kernel_size=63, padding=31, bias=False)
        with torch.no_grad():
            vocoder.subband_upsample.weight.copy_(state_dict["generator.vocoder.subband_upsample.weight"])
            vocoder.synthesis_filter.weight.copy_(state_dict["generator.vocoder.synthesis_filter.weight"])
    return model


def elapsed_ms(start: float) -> float:
    return (perf_counter() - start) * 1000.0


def _edge_right_context(x: torch.Tensor, pad: int) -> torch.Tensor:
    if pad == 0:
        return x[..., :0]
    return x[..., -1:].expand(-1, -1, pad)


def _edge_left_context(x: torch.Tensor, pad: int) -> torch.Tensor:
    if pad == 0:
        return x[..., :0]
    return x[..., :1].expand(-1, -1, pad)


def _reflect_right_context(x: torch.Tensor, pad: int) -> torch.Tensor:
    if pad == 0:
        return x[..., :0]
    reflected = x[..., :-1].flip(-1)
    fallback = x[..., -1:].expand(-1, -1, pad)
    return torch.cat([reflected.repeat(1, 1, pad), fallback], dim=-1)[..., :pad]


def _reflect_left_context(x: torch.Tensor, pad: int) -> torch.Tensor:
    if pad == 0:
        return x[..., :0]
    reflected = x[..., 1:].flip(-1)
    fallback = x[..., :1].expand(-1, -1, pad)
    return torch.cat([reflected.repeat(1, 1, pad), fallback], dim=-1)[..., :pad]


def _select_initial_context(cache: torch.Tensor, initial: torch.Tensor, is_first: torch.Tensor) -> torch.Tensor:
    first = is_first.to(dtype=cache.dtype).reshape(1, 1, 1)
    return cache * (1.0 - first) + initial * first


def _padded_conv_stream_edge(
    module,
    x: torch.Tensor,
    cache: torch.Tensor,
    commit_len: torch.Tensor,
    is_first: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pad = module.pad
    if pad == 0:
        return module.conv(x), x[..., :0]
    left = _select_initial_context(cache, _edge_left_context(x, pad), is_first)
    right = _edge_right_context(x, pad)
    padded = torch.cat([left, x, right], dim=-1)
    history = torch.cat([left, x[..., :commit_len]], dim=-1)
    return module.conv(padded), history[..., -pad:]


def _padded_conv_stream_reflect(
    module,
    x: torch.Tensor,
    cache: torch.Tensor,
    commit_len: torch.Tensor,
    is_first: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pad = module.pad
    if pad == 0:
        return module.conv(x), x[..., :0]
    left = _select_initial_context(cache, _reflect_left_context(x, pad), is_first)
    right = _reflect_right_context(x, pad)
    padded = torch.cat([left, x, right], dim=-1)
    history = torch.cat([left, x[..., :commit_len]], dim=-1)
    return module.conv(padded), history[..., -pad:]


def _residual_stream(
    block,
    x: torch.Tensor,
    cache: torch.Tensor,
    commit_len: torch.Tensor,
    is_first: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    y, next_cache = _padded_conv_stream_reflect(block.conv, block.activation(x), cache, commit_len, is_first)
    return x + block.scale * y, next_cache


def _deconv_stream(
    module,
    x: torch.Tensor,
    cache: torch.Tensor,
    context: int,
    commit_len: torch.Tensor,
    is_first: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined = torch.cat([cache, x], dim=-1)
    cached_y = module(combined)[..., cache.shape[-1] * module.stride[0] :]
    first_y = module(x)
    first = is_first.to(dtype=cached_y.dtype).reshape(1, 1, 1)
    y = cached_y * (1.0 - first) + first_y * first
    history = torch.cat([cache, x[..., :commit_len]], dim=-1)
    return y, history[..., -context:]


class NanoDecoderStreamingWrapper(nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.decoder = generator.decoder
        self.dec_proj = generator.dec_proj
        self.block_context = 3

    def forward(
        self,
        decoder_inputs_context,
        padding_mask_context,
        output_len,
        state_len,
        h1,
        c1,
        h2,
        c2,
        block1_cache,
        block1_cache_mask,
    ):
        output_len = output_len.to(dtype=torch.long)
        state_len = state_len.to(dtype=torch.long)
        context = self.block_context

        block1 = self.decoder.blocks[0]
        block2 = self.decoder.blocks[1]

        valid_mask = ~padding_mask_context
        x = decoder_inputs_context * valid_mask.to(dtype=decoder_inputs_context.dtype).unsqueeze(-1)

        # Block 1: compute outputs for emitted frames plus the right context needed by block 2.
        block1_input_len = output_len + context
        block1_conv_full = self._conv_branch(block1, x, valid_mask)
        block1_conv = block1_conv_full[:, context : context + block1_input_len, :]

        block1_emit_input = x[:, context : context + state_len, :]
        block1_emit_rnn, (h1_new, c1_new) = block1.recurrent(block1_emit_input, (h1, c1))

        block1_right_input = x[:, context + state_len : context + block1_input_len, :]
        block1_right_rnn, _ = block1.recurrent(block1_right_input, (h1_new, c1_new))
        block1_rnn = torch.cat([block1_emit_rnn, block1_right_rnn], dim=1)
        if block1.recurrent_tile != 1:
            block1_rnn = block1_rnn.repeat(1, 1, block1.recurrent_tile)

        block1_residual = x[:, context : context + block1_input_len, :] * block1.residual_scale
        block1_current = block1_conv + block1_rnn + block1_residual
        block1_current = block1_current * valid_mask[:, context : context + block1_input_len].to(
            dtype=block1_current.dtype
        ).unsqueeze(-1)

        block1_current_mask = valid_mask[:, context : context + block1_input_len]
        block1_for_block2 = torch.cat([block1_cache, block1_current], dim=1)
        block1_for_block2_mask = torch.cat([block1_cache_mask, block1_current_mask], dim=1)

        # Block 2: its conv sees cached block-1 left context; its recurrent state advances through emit only.
        block2_conv_full = self._conv_branch(block2, block1_for_block2, block1_for_block2_mask)
        block2_conv = block2_conv_full[:, context : context + output_len, :]

        block2_emit_input = block1_current[:, :state_len, :]
        block2_emit_rnn, (h2_new, c2_new) = block2.recurrent(block2_emit_input, (h2, c2))

        block2_right_input = block1_current[:, state_len:output_len, :]
        block2_right_rnn, _ = block2.recurrent(block2_right_input, (h2_new, c2_new))
        block2_rnn = torch.cat([block2_emit_rnn, block2_right_rnn], dim=1)
        if block2.recurrent_tile != 1:
            block2_rnn = block2_rnn.repeat(1, 1, block2.recurrent_tile)

        block2_residual = block1_current[:, :output_len, :] * block2.residual_scale
        decoded_context = block2_conv + block2_rnn + block2_residual
        decoded_context = decoded_context * valid_mask[:, context : context + output_len].to(
            dtype=decoded_context.dtype
        ).unsqueeze(-1)

        block1_cache_new = torch.cat([block1_cache, block1_current[:, :state_len, :]], dim=1)[:, -context:, :]
        block1_cache_mask_new = torch.cat([block1_cache_mask, block1_current_mask[:, :state_len]], dim=1)[
            :, -context:
        ]
        acoustic = self.dec_proj(decoded_context).transpose(1, 2)
        return acoustic, h1_new, c1_new, h2_new, c2_new, block1_cache_new, block1_cache_mask_new

    @staticmethod
    def _conv_branch(block, x, valid_mask):
        y = x.transpose(1, 2)
        for conv in block.conv_stack:
            y = conv(y)
            y = y * valid_mask.to(dtype=y.dtype).unsqueeze(1)
        return y.transpose(1, 2)


class NanoEncoderWrapper(nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, x, x_lengths, scales):
        generator = self.generator
        d_factor = scales[0]
        p_factor = scales[1]

        x_max_length = x_lengths.max()
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x_max_length), 1).to(x.dtype)
        input_padding_mask = ~x_mask.squeeze(1).bool()

        x, _ = generator.text_embedding(x)
        x = generator.encoder(x, input_padding_mask)

        x, pitch = generator.pitch_predictor.infer(x, input_padding_mask, p_factor)

        durations = generator.duration_predictor.infer(x, input_padding_mask, factor=d_factor)
        y_lengths = durations.sum(dim=1)
        durations = durations.to(torch.long)
        ends = torch.cumsum(durations, dim=1)
        frame_positions = torch.arange(y_lengths.max(), device=x.device, dtype=ends.dtype)

        cumulative_mask = (frame_positions.view(1, 1, -1) < ends.unsqueeze(-1)).to(x.dtype)
        previous_mask = torch.cat(
            [torch.zeros_like(cumulative_mask[:, :1, :]), cumulative_mask[:, :-1, :]],
            dim=1,
        )
        alignment = (cumulative_mask * (1.0 - previous_mask)).transpose(1, 2)
        smoothed_alignment = generator.feature_upsampler.alignment_smoother(alignment.unsqueeze(1)).squeeze(1)
        frames = torch.matmul(smoothed_alignment, x)

        forward_cumsum = torch.cumsum(alignment, dim=1)
        forward_prev = torch.cat([torch.zeros_like(forward_cumsum[:, :1, :]), forward_cumsum[:, :-1, :]], dim=1)
        forward_index = (forward_prev * alignment).sum(dim=-1).to(torch.long)

        reversed_alignment = torch.flip(alignment, dims=(1,))
        backward_cumsum = torch.cumsum(reversed_alignment, dim=1)
        backward_prev = torch.cat([torch.zeros_like(backward_cumsum[:, :1, :]), backward_cumsum[:, :-1, :]], dim=1)
        backward_index = torch.flip(backward_prev * reversed_alignment, dims=(1,)).sum(dim=-1).to(torch.long)

        forward_index = forward_index.clamp_max(generator.feature_upsampler.forward_position.num_embeddings - 1)
        backward_index = backward_index.clamp_max(generator.feature_upsampler.backward_position.num_embeddings - 1)
        decoder_inputs = (
            frames
            + generator.feature_upsampler.forward_position(forward_index)
            + generator.feature_upsampler.backward_position(backward_index)
        )
        target_padding_mask = frame_positions.unsqueeze(0) >= y_lengths.unsqueeze(1)
        return decoder_inputs, target_padding_mask, durations, y_lengths, pitch


class NanoVocoderStreamingWrapper(nn.Module):
    def __init__(self, vocoder):
        super().__init__()
        self.vocoder = vocoder

    def forward(
        self,
        mel_context,
        emit_len,
        is_first,
        gru0,
        gru1,
        prenet_cache,
        stage0_deconv_cache,
        stage0_res0_cache,
        stage0_res1_cache,
        stage0_res2_cache,
        stage1_deconv_cache,
        stage1_res0_cache,
        stage1_res1_cache,
        stage1_res2_cache,
        stage2_deconv_cache,
        stage2_res0_cache,
        stage2_res1_cache,
        stage2_res2_cache,
        stage2_res3_cache,
        stage2_res4_cache,
        final_feature_cache,
        synthesis_cache,
    ):
        emit_len = emit_len.to(dtype=torch.long)
        (
            wave_context,
            gru0_new,
            gru1_new,
            prenet_cache,
            stage0_deconv_cache,
            stage0_res0_cache,
            stage0_res1_cache,
            stage0_res2_cache,
            stage1_deconv_cache,
            stage1_res0_cache,
            stage1_res1_cache,
            stage1_res2_cache,
            stage2_deconv_cache,
            stage2_res0_cache,
            stage2_res1_cache,
            stage2_res2_cache,
            stage2_res3_cache,
            stage2_res4_cache,
            final_feature_cache,
            synthesis_cache,
        ) = self._forward_stream(
            mel_context,
            emit_len,
            is_first,
            gru0,
            gru1,
            prenet_cache,
            stage0_deconv_cache,
            stage0_res0_cache,
            stage0_res1_cache,
            stage0_res2_cache,
            stage1_deconv_cache,
            stage1_res0_cache,
            stage1_res1_cache,
            stage1_res2_cache,
            stage2_deconv_cache,
            stage2_res0_cache,
            stage2_res1_cache,
            stage2_res2_cache,
            stage2_res3_cache,
            stage2_res4_cache,
            final_feature_cache,
            synthesis_cache,
        )
        wave = wave_context[:, :, : emit_len * 300]
        return (
            wave,
            gru0_new,
            gru1_new,
            prenet_cache,
            stage0_deconv_cache,
            stage0_res0_cache,
            stage0_res1_cache,
            stage0_res2_cache,
            stage1_deconv_cache,
            stage1_res0_cache,
            stage1_res1_cache,
            stage1_res2_cache,
            stage2_deconv_cache,
            stage2_res0_cache,
            stage2_res1_cache,
            stage2_res2_cache,
            stage2_res3_cache,
            stage2_res4_cache,
            final_feature_cache,
            synthesis_cache,
        )

    def _forward_stream(
        self,
        mel,
        emit_len,
        is_first,
        gru0,
        gru1,
        prenet_cache,
        stage0_deconv_cache,
        stage0_res0_cache,
        stage0_res1_cache,
        stage0_res2_cache,
        stage1_deconv_cache,
        stage1_res0_cache,
        stage1_res1_cache,
        stage1_res2_cache,
        stage2_deconv_cache,
        stage2_res0_cache,
        stage2_res1_cache,
        stage2_res2_cache,
        stage2_res3_cache,
        stage2_res4_cache,
        final_feature_cache,
        synthesis_cache,
    ):
        vocoder = self.vocoder
        stage0_commit_len = emit_len * 3
        stage1_commit_len = emit_len * 15
        stage2_commit_len = emit_len * 75
        sample_commit_len = emit_len * 300

        x, prenet_cache = _padded_conv_stream_edge(vocoder.prenet[0], mel, prenet_cache, emit_len, is_first)
        x = vocoder.prenet[1](x)
        gru0_emit_out, gru0_new = vocoder.gru0(x[:, :, :emit_len].transpose(1, 2), gru0.unsqueeze(0))
        gru0_right_out, _ = vocoder.gru0(x[:, :, emit_len:].transpose(1, 2), gru0_new)
        gru0_out = torch.cat([gru0_emit_out, gru0_right_out], dim=1)
        x = x + gru0_out.repeat(1, 1, 2).transpose(1, 2)

        x, stage0_deconv_cache = _deconv_stream(
            vocoder.stage0.upsample,
            x,
            stage0_deconv_cache,
            2,
            emit_len,
            is_first,
        )
        x, stage0_res0_cache = _residual_stream(
            vocoder.stage0.residual_blocks[0],
            x,
            stage0_res0_cache,
            stage0_commit_len,
            is_first,
        )
        x, stage0_res1_cache = _residual_stream(
            vocoder.stage0.residual_blocks[1],
            x,
            stage0_res1_cache,
            stage0_commit_len,
            is_first,
        )
        x, stage0_res2_cache = _residual_stream(
            vocoder.stage0.residual_blocks[2],
            x,
            stage0_res2_cache,
            stage0_commit_len,
            is_first,
        )

        x, stage1_deconv_cache = _deconv_stream(
            vocoder.stage1.upsample,
            F.leaky_relu(x, 0.2),
            stage1_deconv_cache,
            2,
            stage0_commit_len,
            is_first,
        )
        x, stage1_res0_cache = _residual_stream(
            vocoder.stage1.residual_blocks[0],
            x,
            stage1_res0_cache,
            stage1_commit_len,
            is_first,
        )
        x, stage1_res1_cache = _residual_stream(
            vocoder.stage1.residual_blocks[1],
            x,
            stage1_res1_cache,
            stage1_commit_len,
            is_first,
        )
        x, stage1_res2_cache = _residual_stream(
            vocoder.stage1.residual_blocks[2],
            x,
            stage1_res2_cache,
            stage1_commit_len,
            is_first,
        )

        gru1_emit_out, gru1_new = vocoder.gru1(x[:, :, :stage1_commit_len].transpose(1, 2), gru1.unsqueeze(0))
        gru1_right_out, _ = vocoder.gru1(x[:, :, stage1_commit_len:].transpose(1, 2), gru1_new)
        gru1_out = torch.cat([gru1_emit_out, gru1_right_out], dim=1)
        x = x + gru1_out.transpose(1, 2)

        x, stage2_deconv_cache = _deconv_stream(
            vocoder.stage2.upsample,
            F.leaky_relu(x, 0.2),
            stage2_deconv_cache,
            2,
            stage1_commit_len,
            is_first,
        )
        x, stage2_res0_cache = _residual_stream(
            vocoder.stage2.residual_blocks[0],
            x,
            stage2_res0_cache,
            stage2_commit_len,
            is_first,
        )
        x, stage2_res1_cache = _residual_stream(
            vocoder.stage2.residual_blocks[1],
            x,
            stage2_res1_cache,
            stage2_commit_len,
            is_first,
        )
        x, stage2_res2_cache = _residual_stream(
            vocoder.stage2.residual_blocks[2],
            x,
            stage2_res2_cache,
            stage2_commit_len,
            is_first,
        )
        x, stage2_res3_cache = _residual_stream(
            vocoder.stage2.residual_blocks[3],
            x,
            stage2_res3_cache,
            stage2_commit_len,
            is_first,
        )
        x, stage2_res4_cache = _residual_stream(
            vocoder.stage2.residual_blocks[4],
            x,
            stage2_res4_cache,
            stage2_commit_len,
            is_first,
        )

        x, final_feature_cache = _padded_conv_stream_edge(
            vocoder.final_feature_conv,
            F.leaky_relu(x, 0.2),
            final_feature_cache,
            stage2_commit_len,
            is_first,
        )
        x = F.leaky_relu(x, 0.2)

        bands = [head(x.transpose(1, 2)).transpose(1, 2) for head in vocoder.band_heads]
        bands = torch.cat(bands, dim=1)
        if hasattr(vocoder, "subband_upsample"):
            subband_cache = synthesis_cache[..., :1]
            filter_cache = synthesis_cache[..., 1:]
            pqmf_bands, subband_cache = _deconv_stream(
                vocoder.subband_upsample,
                bands,
                subband_cache,
                1,
                stage2_commit_len,
                is_first,
            )
            synthesis_weight = vocoder.synthesis_filter.weight
            synthesis_bias = vocoder.synthesis_filter.bias
        else:
            filter_cache = synthesis_cache
            pqmf_bands = F.conv_transpose1d(
                bands,
                vocoder.pqmf.updown_filter * vocoder.pqmf.subbands,
                stride=vocoder.pqmf.subbands,
            )
            synthesis_weight = vocoder.pqmf.synthesis_filter
            synthesis_bias = None
        padded = torch.cat(
            [
                filter_cache,
                pqmf_bands,
                pqmf_bands.new_zeros(pqmf_bands.shape[0], pqmf_bands.shape[1], 31),
            ],
            dim=-1,
        )
        wave = F.conv1d(padded, synthesis_weight, synthesis_bias)
        filter_cache = torch.cat([filter_cache, pqmf_bands[..., :sample_commit_len]], dim=-1)[..., -31:]
        synthesis_cache = (
            torch.cat([subband_cache, filter_cache], dim=-1)
            if hasattr(vocoder, "subband_upsample")
            else filter_cache
        )

        return (
            wave,
            gru0_new.squeeze(0),
            gru1_new.squeeze(0),
            prenet_cache,
            stage0_deconv_cache,
            stage0_res0_cache,
            stage0_res1_cache,
            stage0_res2_cache,
            stage1_deconv_cache,
            stage1_res0_cache,
            stage1_res1_cache,
            stage1_res2_cache,
            stage2_deconv_cache,
            stage2_res0_cache,
            stage2_res1_cache,
            stage2_res2_cache,
            stage2_res3_cache,
            stage2_res4_cache,
            final_feature_cache,
            synthesis_cache,
        )


ENCODER_INPUT_NAMES = ["x", "x_lengths", "scales"]
ENCODER_OUTPUT_NAMES = ["decoder_inputs", "target_padding_mask", "durations", "y_lengths", "pitch"]

DECODER_INPUT_NAMES = [
    "decoder_inputs_context",
    "padding_mask_context",
    "output_len",
    "state_len",
    "h1",
    "c1",
    "h2",
    "c2",
    "block1_cache",
    "block1_cache_mask",
]
DECODER_OUTPUT_NAMES = [
    "acoustic",
    "h1_out",
    "c1_out",
    "h2_out",
    "c2_out",
    "block1_cache_out",
    "block1_cache_mask_out",
]

VOCODER_STATE_NAMES = [
    "gru0",
    "gru1",
    "prenet_cache",
    "stage0_deconv_cache",
    "stage0_res0_cache",
    "stage0_res1_cache",
    "stage0_res2_cache",
    "stage1_deconv_cache",
    "stage1_res0_cache",
    "stage1_res1_cache",
    "stage1_res2_cache",
    "stage2_deconv_cache",
    "stage2_res0_cache",
    "stage2_res1_cache",
    "stage2_res2_cache",
    "stage2_res3_cache",
    "stage2_res4_cache",
    "final_feature_cache",
    "synthesis_cache",
]
VOCODER_INPUT_NAMES = ["mel_context", "emit_len", "is_first", *VOCODER_STATE_NAMES]
VOCODER_OUTPUT_NAMES = ["wave", *[f"{name}_out" for name in VOCODER_STATE_NAMES]]


def initial_decoder_state(batch_size=1):
    return {
        "h1": np.zeros((1, batch_size, 128), dtype=np.float32),
        "c1": np.zeros((1, batch_size, 128), dtype=np.float32),
        "h2": np.zeros((1, batch_size, 128), dtype=np.float32),
        "c2": np.zeros((1, batch_size, 128), dtype=np.float32),
        "block1_cache": np.zeros((batch_size, 3, 256), dtype=np.float32),
        "block1_cache_mask": np.zeros((batch_size, 3), dtype=bool),
    }


def initial_vocoder_state(batch_size=1, input_channels=128, legacy_synthesis=False):
    zeros = np.zeros
    return {
        "gru0": zeros((batch_size, 128), dtype=np.float32),
        "gru1": zeros((batch_size, 64), dtype=np.float32),
        "prenet_cache": zeros((batch_size, input_channels, 3), dtype=np.float32),
        "stage0_deconv_cache": zeros((batch_size, 256, 2), dtype=np.float32),
        "stage0_res0_cache": zeros((batch_size, 128, 1), dtype=np.float32),
        "stage0_res1_cache": zeros((batch_size, 128, 3), dtype=np.float32),
        "stage0_res2_cache": zeros((batch_size, 128, 9), dtype=np.float32),
        "stage1_deconv_cache": zeros((batch_size, 128, 2), dtype=np.float32),
        "stage1_res0_cache": zeros((batch_size, 64, 1), dtype=np.float32),
        "stage1_res1_cache": zeros((batch_size, 64, 3), dtype=np.float32),
        "stage1_res2_cache": zeros((batch_size, 64, 9), dtype=np.float32),
        "stage2_deconv_cache": zeros((batch_size, 64, 2), dtype=np.float32),
        "stage2_res0_cache": zeros((batch_size, 32, 1), dtype=np.float32),
        "stage2_res1_cache": zeros((batch_size, 32, 3), dtype=np.float32),
        "stage2_res2_cache": zeros((batch_size, 32, 9), dtype=np.float32),
        "stage2_res3_cache": zeros((batch_size, 32, 27), dtype=np.float32),
        "stage2_res4_cache": zeros((batch_size, 32, 81), dtype=np.float32),
        "final_feature_cache": zeros((batch_size, 32, 3), dtype=np.float32),
        "synthesis_cache": zeros((batch_size, 4, 32 if legacy_synthesis else 31), dtype=np.float32),
    }


def export_nano_streaming(
    model,
    output_dir: Path,
    opset: int,
    chunk_frames: int,
    frontend_inputs,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = model.generator.eval()
    encoder_path = output_dir / "nano_encoder.onnx"
    decoder_path = output_dir / "nano_decoder_stream.onnx"
    vocoder_path = output_dir / "nano_vocoder_stream.onnx"

    x, x_lengths, scales = frontend_inputs

    encoder = NanoEncoderWrapper(generator).eval()
    torch.onnx.export(
        encoder,
        (x, x_lengths, scales),
        encoder_path,
        input_names=ENCODER_INPUT_NAMES,
        output_names=ENCODER_OUTPUT_NAMES,
        dynamic_axes={
            "x": {0: "batch", 1: "tokens"},
            "x_lengths": {0: "batch"},
            "decoder_inputs": {0: "batch", 1: "frames"},
            "target_padding_mask": {0: "batch", 1: "frames"},
            "durations": {0: "batch", 1: "tokens"},
            "y_lengths": {0: "batch"},
            "pitch": {0: "batch", 1: "tokens"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(encoder_path))

    decoder = NanoDecoderStreamingWrapper(generator).eval()
    decoder_args = (
        torch.zeros(1, chunk_frames + 21, model.hparams.dim, dtype=torch.float32),
        torch.ones(1, chunk_frames + 21, dtype=torch.bool),
        torch.tensor(chunk_frames + 12, dtype=torch.long),
        torch.tensor(chunk_frames, dtype=torch.long),
        torch.zeros(1, 1, 128, dtype=torch.float32),
        torch.zeros(1, 1, 128, dtype=torch.float32),
        torch.zeros(1, 1, 128, dtype=torch.float32),
        torch.zeros(1, 1, 128, dtype=torch.float32),
        torch.zeros(1, 3, model.hparams.dim, dtype=torch.float32),
        torch.zeros(1, 3, dtype=torch.bool),
    )
    torch.onnx.export(
        decoder,
        decoder_args,
        decoder_path,
        input_names=DECODER_INPUT_NAMES,
        output_names=DECODER_OUTPUT_NAMES,
        dynamic_axes={
            "decoder_inputs_context": {0: "batch", 1: "context_frames"},
            "padding_mask_context": {0: "batch", 1: "context_frames"},
            "block1_cache": {0: "batch"},
            "block1_cache_mask": {0: "batch"},
            "acoustic": {0: "batch", 2: "emit_frames"},
            "block1_cache_out": {0: "batch"},
            "block1_cache_mask_out": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(decoder_path))

    vocoder = NanoVocoderStreamingWrapper(generator.vocoder).eval()
    input_channels = int(generator.dec_proj.out_features)
    legacy_synthesis = hasattr(generator.vocoder, "subband_upsample")
    vocoder_state = initial_vocoder_state(
        input_channels=input_channels,
        legacy_synthesis=legacy_synthesis,
    )
    vocoder_args = [
        torch.zeros(1, input_channels, chunk_frames + 12, dtype=torch.float32),
        torch.tensor(chunk_frames, dtype=torch.long),
        torch.tensor(True, dtype=torch.bool),
    ]
    vocoder_args.extend(torch.from_numpy(vocoder_state[name]) for name in VOCODER_STATE_NAMES)
    torch.onnx.export(
        vocoder,
        tuple(vocoder_args),
        vocoder_path,
        input_names=VOCODER_INPUT_NAMES,
        output_names=VOCODER_OUTPUT_NAMES,
        dynamic_axes={
            "mel_context": {0: "batch", 2: "context_frames"},
            "wave": {0: "batch", 2: "samples"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(vocoder_path))
    return encoder_path, decoder_path, vocoder_path


def _int8_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_int8{path.suffix}")


def quantize_encoder_decoder(encoder_path: Path, decoder_path: Path) -> tuple[Path, Path]:
    quantized_encoder_path = _int8_path(encoder_path)
    quantized_decoder_path = _int8_path(decoder_path)
    quantize_dynamic(
        str(encoder_path),
        str(quantized_encoder_path),
        op_types_to_quantize=QUANTIZED_OP_TYPES,
        weight_type=QuantType.QInt8,
    )
    quantize_dynamic(
        str(decoder_path),
        str(quantized_decoder_path),
        op_types_to_quantize=QUANTIZED_OP_TYPES,
        weight_type=QuantType.QInt8,
    )
    onnx.checker.check_model(onnx.load(quantized_encoder_path))
    onnx.checker.check_model(onnx.load(quantized_decoder_path))
    return quantized_encoder_path, quantized_decoder_path


@torch.inference_mode()
def prepare_token_inputs(model, text, d_factor, p_factor):
    inputs = model.prepare_input(
        text,
        d_factor=d_factor,
        p_factor=p_factor,
        split_sentences=False,
    )
    scales = torch.tensor([d_factor, p_factor], dtype=torch.float32)
    return inputs, (inputs.x, inputs.x_lengths, scales)


def make_session(path: Path, intra_op_num_threads: int | None = None) -> ort.InferenceSession:
    options = ort.SessionOptions()
    if intra_op_num_threads is not None:
        options.intra_op_num_threads = intra_op_num_threads
    return ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])


@torch.inference_mode()
def run_pytorch_encoder_reference(model, frontend_inputs):
    _inputs, (x, x_lengths, scales) = frontend_inputs
    generator = model.generator
    d_factor = scales[0]
    p_factor = scales[1]

    x_max_length = x_lengths.max()
    x_mask = torch.unsqueeze(sequence_mask(x_lengths, x_max_length), 1).to(x.dtype)
    input_padding_mask = ~x_mask.squeeze(1).bool()

    encoded, _ = generator.text_embedding(x)
    encoded = generator.encoder(encoded, input_padding_mask)

    encoded, pitch = generator.pitch_predictor.infer(encoded, input_padding_mask, p_factor)

    durations = generator.duration_predictor.infer(encoded, input_padding_mask, factor=d_factor)
    y_lengths = durations.sum(dim=1)
    y_max_length = y_lengths.max()
    y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).type_as(encoded)
    target_padding_mask = ~y_mask.squeeze(1).bool()
    decoder_inputs = generator.feature_upsampler(encoded, durations)
    return {
        "decoder_inputs": decoder_inputs.detach().cpu().numpy(),
        "target_padding_mask": target_padding_mask.detach().cpu().numpy(),
        "durations": durations.detach().cpu().numpy(),
        "y_lengths": y_lengths.detach().cpu().numpy(),
        "pitch": pitch.detach().cpu().numpy(),
    }


def check_encoder_parity(model, encoder_session, frontend_inputs, strict: bool = True):
    reference = run_pytorch_encoder_reference(model, frontend_inputs)
    ort_outputs = run_ort_encoder(encoder_session, frontend_inputs)
    keys = ("decoder_inputs", "target_padding_mask", "durations", "y_lengths", "pitch")
    for key in keys:
        if reference[key].shape != ort_outputs[key].shape:
            message = f"{key} shape mismatch: {reference[key].shape} != {ort_outputs[key].shape}"
            if strict:
                raise AssertionError(message)
            return {"ok": False, "reason": message}
    if not np.array_equal(reference["durations"], ort_outputs["durations"]):
        message = "duration mismatch between PyTorch and ONNX encoder"
        if strict:
            raise AssertionError(message)
        return {"ok": False, "reason": message}
    if not np.array_equal(reference["y_lengths"], ort_outputs["y_lengths"]):
        message = "y_lengths mismatch between PyTorch and ONNX encoder"
        if strict:
            raise AssertionError(message)
        return {"ok": False, "reason": message}
    if not np.array_equal(reference["target_padding_mask"], ort_outputs["target_padding_mask"]):
        message = "target_padding_mask mismatch between PyTorch and ONNX encoder"
        if strict:
            raise AssertionError(message)
        return {"ok": False, "reason": message}

    decoder_inputs_abs = np.max(np.abs(reference["decoder_inputs"] - ort_outputs["decoder_inputs"]))
    pitch_abs = np.max(np.abs(reference["pitch"] - ort_outputs["pitch"]))
    if decoder_inputs_abs > 2e-4 or pitch_abs > 2e-4:
        message = (
            "encoder parity failed: "
            f"decoder_inputs_abs={decoder_inputs_abs:.6g}, "
            f"pitch_abs={pitch_abs:.6g}"
        )
        if strict:
            raise AssertionError(message)
        return {"ok": False, "reason": message}
    return {
        "ok": True,
        "reason": "",
        "decoder_inputs_max_abs": float(decoder_inputs_abs),
        "pitch_max_abs": float(pitch_abs),
    }


def run_ort_encoder(encoder_session, frontend_inputs):
    inputs, (x, x_lengths, scales) = frontend_inputs
    feed = {
        "x": x.detach().cpu().numpy().astype(np.int64),
        "x_lengths": x_lengths.detach().cpu().numpy().astype(np.int64),
        "scales": scales.detach().cpu().numpy().astype(np.float32),
    }
    t0 = perf_counter()
    decoder_inputs, target_padding_mask, durations, y_lengths, pitch = encoder_session.run(None, feed)
    encoder_ms = elapsed_ms(t0)
    return {
        "inputs": inputs,
        "decoder_inputs": decoder_inputs,
        "target_padding_mask": target_padding_mask,
        "durations": durations,
        "y_lengths": y_lengths,
        "pitch": pitch,
        "encoder_ms": encoder_ms,
    }


def _pad_time_axis(x: np.ndarray, target_frames: int, axis: int, pad_value=0) -> np.ndarray:
    current_frames = x.shape[axis]
    if current_frames >= target_frames:
        index = [slice(None)] * x.ndim
        index[axis] = slice(0, target_frames)
        return x[tuple(index)]
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, target_frames - current_frames)
    return np.pad(x, pad_width, mode="constant", constant_values=pad_value)


def _right_pad_features(x: np.ndarray, target_frames: int, axis: int) -> np.ndarray:
    current_frames = x.shape[axis]
    if current_frames >= target_frames:
        index = [slice(None)] * x.ndim
        index[axis] = slice(0, target_frames)
        return x[tuple(index)]
    if current_frames == 0:
        return _pad_time_axis(x, target_frames, axis, pad_value=0)
    last_index = [slice(None)] * x.ndim
    last_index[axis] = slice(current_frames - 1, current_frames)
    repeat = np.repeat(x[tuple(last_index)], target_frames - current_frames, axis=axis)
    return np.concatenate([x, repeat], axis=axis)


def _context_window(
    x: np.ndarray,
    start: int,
    end: int,
    left_context: int,
    right_context: int,
    axis: int,
    *,
    pad_value=0,
    edge_pad: bool = True,
) -> tuple[np.ndarray, int]:
    total_frames = x.shape[axis]
    context_start = max(0, start - left_context)
    context_end = min(total_frames, end + right_context)
    left_pad = start - left_context - context_start
    right_pad = end + right_context - context_end

    index = [slice(None)] * x.ndim
    index[axis] = slice(context_start, context_end)
    window = x[tuple(index)]

    if left_pad < 0:
        if edge_pad and window.shape[axis] > 0:
            first_index = [slice(None)] * window.ndim
            first_index[axis] = slice(0, 1)
            pad = np.repeat(window[tuple(first_index)], -left_pad, axis=axis)
        else:
            shape = list(window.shape)
            shape[axis] = -left_pad
            pad = np.full(shape, pad_value, dtype=window.dtype)
        window = np.concatenate([pad, window], axis=axis)

    if right_pad > 0:
        if edge_pad and window.shape[axis] > 0:
            last_index = [slice(None)] * window.ndim
            last_index[axis] = slice(window.shape[axis] - 1, window.shape[axis])
            pad = np.repeat(window[tuple(last_index)], right_pad, axis=axis)
        else:
            shape = list(window.shape)
            shape[axis] = right_pad
            pad = np.full(shape, pad_value, dtype=window.dtype)
        window = np.concatenate([window, pad], axis=axis)

    emit_offset = left_context
    return window, emit_offset


def _run_decoder(decoder_session, decoder_inputs_context, padding_mask_context, output_len, state_len, decoder_state):
    decoder_outs = decoder_session.run(
        None,
        {
            "decoder_inputs_context": np.asarray(decoder_inputs_context, dtype=np.float32),
            "padding_mask_context": np.asarray(padding_mask_context, dtype=bool),
            "output_len": np.asarray(output_len, dtype=np.int64),
            "state_len": np.asarray(state_len, dtype=np.int64),
            **decoder_state,
        },
    )
    return decoder_outs[0], dict(zip(["h1", "c1", "h2", "c2", "block1_cache", "block1_cache_mask"], decoder_outs[1:]))


def _run_vocoder(vocoder_session, mel_context, emit_len, is_first, vocoder_state):
    vocoder_outs = vocoder_session.run(
        None,
        {
            "mel_context": np.asarray(mel_context, dtype=np.float32),
            "emit_len": np.asarray(emit_len, dtype=np.int64),
            "is_first": np.asarray(is_first, dtype=bool),
            **vocoder_state,
        },
    )
    return vocoder_outs[0], dict(zip(VOCODER_STATE_NAMES, vocoder_outs[1:]))


def run_ort_stream(
    decoder_session,
    vocoder_session,
    decoder_inputs,
    target_padding_mask,
    chunk_frames,
    decoder_rf_frames,
    vocoder_rf_frames,
    return_wave: bool = False,
    return_acoustic: bool = False,
):
    decoder_inputs_np = np.asarray(decoder_inputs, dtype=np.float32)
    target_padding_mask_np = np.asarray(target_padding_mask, dtype=bool)
    decoder_state = initial_decoder_state(batch_size=decoder_inputs_np.shape[0])
    synthesis_cache_input = next(item for item in vocoder_session.get_inputs() if item.name == "synthesis_cache")
    vocoder_state = initial_vocoder_state(
        batch_size=decoder_inputs_np.shape[0],
        input_channels=vocoder_session.get_inputs()[0].shape[1],
        legacy_synthesis=synthesis_cache_input.shape[-1] == 32,
    )

    total_frames = decoder_inputs_np.shape[1]
    required_frames = max(decoder_rf_frames, vocoder_rf_frames)
    consumed_frames = 0
    emitted_samples = 0
    first_api_audio_ms = None
    first_rf_ready_audio_ms = None
    emitted_chunks = []
    acoustic_chunks = []

    t0 = perf_counter()
    decoder_block_context = 3
    decoder_left_context = decoder_block_context
    decoder_right_context = max(decoder_rf_frames, decoder_block_context * 2)
    for start in range(0, total_frames, chunk_frames):
        end = min(start + chunk_frames, total_frames)
        actual_frames = end - start
        future_frames = min(vocoder_rf_frames, total_frames - end)
        vocoder_context_frames = actual_frames + max(future_frames, 1)
        acoustic_output_frames = actual_frames + max(future_frames, 1)
        decoder_context, _ = _context_window(
            decoder_inputs_np,
            start,
            start + acoustic_output_frames,
            decoder_left_context,
            decoder_right_context,
            axis=1,
            pad_value=0,
            edge_pad=False,
        )
        decoder_context_mask, _ = _context_window(
            target_padding_mask_np,
            start,
            start + acoustic_output_frames,
            decoder_left_context,
            decoder_right_context,
            axis=1,
            pad_value=True,
            edge_pad=False,
        )

        acoustic_context, decoder_state = _run_decoder(
            decoder_session,
            decoder_context,
            decoder_context_mask,
            acoustic_output_frames,
            actual_frames,
            decoder_state,
        )
        wave, vocoder_state = _run_vocoder(
            vocoder_session,
            acoustic_context[:, :, :vocoder_context_frames],
            actual_frames,
            start == 0,
            vocoder_state,
        )

        consumed_frames += actual_frames
        emitted_samples += int(wave.shape[-1])
        if return_wave:
            emitted_chunks.append(wave)
        if return_acoustic:
            acoustic_chunks.append(acoustic_context[:, :, :actual_frames])
        if first_api_audio_ms is None and wave.size > 0:
            first_api_audio_ms = elapsed_ms(t0)
        if first_rf_ready_audio_ms is None and consumed_frames >= required_frames and wave.size > 0:
            first_rf_ready_audio_ms = elapsed_ms(t0)

    stream_ms = elapsed_ms(t0)
    result = {
        "total_frames": total_frames,
        "required_frames": required_frames,
        "chunks": int(np.ceil(total_frames / chunk_frames)),
        "first_api_audio_ms": first_api_audio_ms,
        "first_rf_ready_audio_ms": first_rf_ready_audio_ms,
        "stream_ms": stream_ms,
        "emitted_samples": emitted_samples,
    }
    if return_wave:
        result["wave"] = np.concatenate(emitted_chunks, axis=-1) if emitted_chunks else np.zeros(
            (decoder_inputs_np.shape[0], 1, 0),
            dtype=np.float32,
        )
    if return_acoustic:
        result["acoustic"] = np.concatenate(acoustic_chunks, axis=-1) if acoustic_chunks else np.zeros(
            (decoder_inputs_np.shape[0], 128, 0),
            dtype=np.float32,
        )
    return result


@torch.inference_mode()
def run_pytorch_full_decoder_vocoder(model, decoder_inputs, target_padding_mask):
    generator = model.generator
    decoder_inputs_t = torch.from_numpy(np.asarray(decoder_inputs, dtype=np.float32))
    padding_mask_t = torch.from_numpy(np.asarray(target_padding_mask, dtype=bool))
    decoded = generator.decoder(decoder_inputs_t, padding_mask_t)
    acoustic = generator.dec_proj(decoded).transpose(1, 2)
    wave = generator.vocoder(acoustic, None, padding_mask_t)
    if wave.ndim == 2:
        wave = wave.unsqueeze(1)
    return acoustic.detach().cpu().numpy(), wave.detach().cpu().numpy()


def _boundary_error(error, chunk_size):
    if error.shape[-1] <= chunk_size:
        return 0.0, 0.0, 0
    values = []
    radius = min(max(1, chunk_size // 2), 300)
    for boundary in range(chunk_size, error.shape[-1], chunk_size):
        if boundary + chunk_size > error.shape[-1]:
            continue
        left = max(0, boundary - radius)
        right = min(error.shape[-1], boundary + radius)
        values.append(float(np.max(np.abs(error[..., left:right]))))
    return max(values), float(np.mean(values)), len(values)


def _compare_arrays(reference, candidate, chunk_size):
    frames = min(reference.shape[-1], candidate.shape[-1])
    reference = reference[..., :frames]
    candidate = candidate[..., :frames]
    error = candidate - reference
    boundary_max, boundary_mean, boundary_count = _boundary_error(error, chunk_size)
    return {
        "reference_len": int(reference.shape[-1]),
        "candidate_len": int(candidate.shape[-1]),
        "max_abs": float(np.max(np.abs(error))) if frames else 0.0,
        "mean_abs": float(np.mean(np.abs(error))) if frames else 0.0,
        "rms_error": float(np.sqrt(np.mean(error**2))) if frames else 0.0,
        "reference_rms": float(np.sqrt(np.mean(reference**2))) if frames else 0.0,
        "boundary_max_abs": boundary_max,
        "boundary_mean_abs": boundary_mean,
        "boundary_count": boundary_count,
    }


def check_streaming_parity(
    model,
    decoder_session,
    vocoder_session,
    decoder_inputs,
    target_padding_mask,
    chunk_frames,
    decoder_rf_frames,
    vocoder_rf_frames,
):
    reference_acoustic, reference_wave = run_pytorch_full_decoder_vocoder(
        model,
        decoder_inputs,
        target_padding_mask,
    )
    stream = run_ort_stream(
        decoder_session,
        vocoder_session,
        decoder_inputs,
        target_padding_mask,
        chunk_frames,
        decoder_rf_frames,
        vocoder_rf_frames,
        return_wave=True,
        return_acoustic=True,
    )
    return {
        "decoder": _compare_arrays(
            reference_acoustic,
            stream["acoustic"],
            chunk_frames,
        ),
        "wave": _compare_arrays(
            reference_wave,
            stream["wave"],
            chunk_frames * 300,
        ),
    }


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024.0 * 1024.0)


def benchmark_pipeline(
    label: str,
    model,
    token_inputs,
    encoder_path: Path,
    decoder_path: Path,
    vocoder_path: Path,
    args,
    *,
    strict_parity: bool,
):
    encoder_session = make_session(encoder_path, args.intra_op_num_threads)
    decoder_session = make_session(decoder_path, args.intra_op_num_threads)
    vocoder_session = make_session(vocoder_path, args.intra_op_num_threads)
    parity = None if args.skip_parity else check_encoder_parity(
        model,
        encoder_session,
        token_inputs,
        strict=strict_parity,
    )
    frontend = run_ort_encoder(encoder_session, token_inputs)
    stream = run_ort_stream(
        decoder_session,
        vocoder_session,
        frontend["decoder_inputs"],
        frontend["target_padding_mask"],
        args.chunk_frames,
        args.decoder_rf_frames,
        args.vocoder_rf_frames,
    )
    streaming_parity = None
    if args.check_stream_parity and strict_parity:
        streaming_parity = check_streaming_parity(
            model,
            decoder_session,
            vocoder_session,
            frontend["decoder_inputs"],
            frontend["target_padding_mask"],
            args.chunk_frames,
            args.decoder_rf_frames,
            args.vocoder_rf_frames,
        )
    audio_ms = (stream["emitted_samples"] / model.sample_rate) * 1000.0
    stream_rtf = stream["stream_ms"] / audio_ms if audio_ms > 0 else float("inf")
    return {
        "label": label,
        "encoder_path": encoder_path,
        "decoder_path": decoder_path,
        "vocoder_path": vocoder_path,
        "parity": parity,
        "frontend": frontend,
        "stream": stream,
        "streaming_parity": streaming_parity,
        "audio_ms": audio_ms,
        "stream_rtf": stream_rtf,
    }


def print_benchmark(result):
    frontend = result["frontend"]
    stream = result["stream"]
    parity = result["parity"]
    streaming_parity = result["streaming_parity"]
    print(f"\n[{result['label']}]")
    print(f"encoder_onnx: {result['encoder_path']} ({file_size_mb(result['encoder_path']):.3f} MiB)")
    print(f"decoder_onnx: {result['decoder_path']} ({file_size_mb(result['decoder_path']):.3f} MiB)")
    print(f"vocoder_onnx: {result['vocoder_path']} ({file_size_mb(result['vocoder_path']):.3f} MiB, f32)")
    print(f"text_frames: {int(frontend['durations'].shape[1])}")
    print(f"acoustic_frames: {stream['total_frames']}")
    print(f"chunks: {stream['chunks']}")
    print(f"emitted_samples: {stream['emitted_samples']}")
    print(f"emitted_audio_ms: {result['audio_ms']:.3f}")
    print(f"encoder_to_decoder_inputs_ms: {frontend['encoder_ms']:.3f}")
    if parity is not None:
        print(f"encoder_parity_ok: {parity['ok']}")
        if parity["ok"]:
            print(f"encoder_parity_decoder_inputs_max_abs: {parity['decoder_inputs_max_abs']:.8f}")
            print(f"encoder_parity_pitch_max_abs: {parity['pitch_max_abs']:.8f}")
        else:
            print(f"encoder_parity_reason: {parity['reason']}")
    print(f"first_api_audio_ms: {stream['first_api_audio_ms']:.3f}")
    print(f"first_rf_ready_audio_ms: {stream['first_rf_ready_audio_ms']:.3f}")
    print(f"stream_total_ms: {stream['stream_ms']:.3f}")
    print(f"stream_rtf: {result['stream_rtf']:.6f}")
    print(f"text_to_first_rf_ready_audio_ms: {frontend['encoder_ms'] + stream['first_rf_ready_audio_ms']:.3f}")
    if streaming_parity is not None:
        for name, metrics in streaming_parity.items():
            print(f"{name}_parity_max_abs: {metrics['max_abs']:.8f}")
            print(f"{name}_parity_mean_abs: {metrics['mean_abs']:.8f}")
            print(f"{name}_parity_rms_error: {metrics['rms_error']:.8f}")
            print(f"{name}_parity_reference_rms: {metrics['reference_rms']:.8f}")
            print(f"{name}_parity_boundary_max_abs: {metrics['boundary_max_abs']:.8f}")
            print(f"{name}_parity_boundary_mean_abs: {metrics['boundary_mean_abs']:.8f}")
            print(f"{name}_parity_boundary_count: {metrics['boundary_count']}")


def main():
    parser = argparse.ArgumentParser(description="Export and benchmark nano decoder/vocoder streaming ONNX on CPU.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("onnx/nano_streaming"))
    parser.add_argument("--text", type=str, default=DEFAULT_TEXT)
    parser.add_argument("--chunk-frames", type=int, default=8)
    parser.add_argument("--decoder-rf-frames", type=int, default=6)
    parser.add_argument("--vocoder-rf-frames", type=int, default=12)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--intra-op-num-threads", type=int, default=None)
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--quantize-encoder-decoder", action="store_true")
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--check-stream-parity", action="store_true")
    parser.add_argument("--d-factor", type=float, default=None)
    parser.add_argument("--p-factor", type=float, default=None)
    args = parser.parse_args()

    checkpoint = args.checkpoint or find_latest_checkpoint(args.runs_dir)
    model = load_model_for_export(checkpoint)
    model.eval()
    model.freeze()

    d_factor = args.d_factor if args.d_factor is not None else model.inference_args.d_factor
    p_factor = args.p_factor if args.p_factor is not None else model.inference_args.p_factor

    if args.no_export:
        encoder_path = args.output_dir / "nano_encoder.onnx"
        decoder_path = args.output_dir / "nano_decoder_stream.onnx"
        vocoder_path = args.output_dir / "nano_vocoder_stream.onnx"
    else:
        token_inputs = prepare_token_inputs(
            model,
            args.text,
            d_factor=d_factor,
            p_factor=p_factor,
        )
        encoder_path, decoder_path, vocoder_path = export_nano_streaming(
            model,
            args.output_dir,
            args.opset,
            args.chunk_frames,
            token_inputs[1],
        )

    token_inputs = prepare_token_inputs(model, args.text, d_factor=d_factor, p_factor=p_factor)
    results = [
        benchmark_pipeline(
            "f32 encoder + f32 decoder + f32 vocoder",
            model,
            token_inputs,
            encoder_path,
            decoder_path,
            vocoder_path,
            args,
            strict_parity=True,
        )
    ]
    if args.quantize_encoder_decoder:
        quantized_encoder_path, quantized_decoder_path = quantize_encoder_decoder(encoder_path, decoder_path)
        results.append(
            benchmark_pipeline(
                "int8 encoder + int8 decoder + f32 vocoder",
                model,
                token_inputs,
                quantized_encoder_path,
                quantized_decoder_path,
                vocoder_path,
                args,
                strict_parity=False,
            )
        )

    print(f"checkpoint: {checkpoint}")
    print("device: cpu")
    print(f"clean_text: {token_inputs[0].clean_text}")
    print(f"chunk_frames: {args.chunk_frames}")
    print(f"decoder_rf_frames: {args.decoder_rf_frames}")
    print(f"vocoder_rf_frames: {args.vocoder_rf_frames}")
    print(f"rf_ready_after_frames: {max(args.decoder_rf_frames, args.vocoder_rf_frames)}")
    for result in results:
        print_benchmark(result)


if __name__ == "__main__":
    main()
