import numpy as np

# MUST happen before Synthesizer is created
from tts.patch_coqui_cleaners import *  # noqa: F401,F403

from TTS.utils.synthesizer import Synthesizer

class CoquiTacotron2TTS:
    def __init__(self, tts_checkpoint: str, tts_config_path: str, device: str = "cpu"):
        self.synth = Synthesizer(
            tts_checkpoint=tts_checkpoint,
            tts_config_path=tts_config_path,
            vocoder_checkpoint="",
            vocoder_config="",
            use_cuda=device.startswith("cuda"),
        )

    def tts_to_wav(self, text: str) -> np.ndarray:
        wav = self.synth.tts(text)
        return np.asarray(wav, dtype=np.float32)

    @property
    def sample_rate(self) -> int:
        return int(self.synth.output_sample_rate)