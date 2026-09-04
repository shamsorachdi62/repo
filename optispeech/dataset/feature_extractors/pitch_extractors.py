import dataclasses
import typing
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

import librosa
import numpy as np
import torch
import torchaudio
try:
    import torchcrepe
except ImportError:
    torchcrepe = None
try:
    import penn
except ImportError:
    penn = None
try:
    import pyworld as pw
except ImportError:
    pw = None
from scipy.interpolate import interp1d

from optispeech.utils import pylogger, trim_or_pad_to_target_length


log = pylogger.get_pylogger(__name__)


@dataclass
class BasePitchExtractor(ABC):
    sample_rate: int
    n_feats: int
    hop_length: int
    n_fft: int
    win_length: int
    f_min: int
    f_max: int
    batch_size: int
    interpolate: bool = True

    def __post_init__(self):
        pass

    @abstractmethod
    def __call__(self, wav: np.ndarray, mel_length: int) -> np.ndarray:
        """Extract pitch."""

    def __getstate__(self):
        return dataclasses.asdict(self)

    def __setstate__(self, state):
        for (attr, value) in state.items():
            setattr(self, attr, value)
        self.__post_init__()

    @staticmethod
    def perform_interpolation(pitch):
        # interpolate to cover the unvoiced segments as well
        nonzero_ids = np.where(pitch != 0)[0]
        interp_fn = interp1d(
            nonzero_ids,
            pitch[nonzero_ids],
            fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
            bounds_error=False,
        )
        pitch = interp_fn(np.arange(0, len(pitch)))
        return pitch


@dataclass
class DIOPitchExtractor(BasePitchExtractor):
    _METHOD: typing.ClassVar[str] = "dio"

    def __post_init__(self):
        self.extraction_func = getattr(pw, self._METHOD)

    def __call__(self, wav, mel_length):
        wav = wav.astype(np.double)
        pitch, t = self.extraction_func(
            wav, self.sample_rate, frame_period=self.hop_length / self.sample_rate * 1000
        )
        pitch = pw.stonemask(wav, pitch, t, self.sample_rate)
        pitch = trim_or_pad_to_target_length(pitch, mel_length)
        if self.interpolate:
            pitch = self.perform_interpolation(pitch)
        return pitch


class HarvestPitchExtractor(DIOPitchExtractor):
    _METHOD: str = "harvest"

