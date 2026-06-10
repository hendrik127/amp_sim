import numpy as np
import soundfile as sf


def load_mono_audio(path):
    """
    Load audio as mono float32.
    """
    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio.astype(np.float32), sr

