import numpy as np
import torch

MIN_SEPARATION_MS = 300
ENERGY_THRESHOLD = 0.4


def detect_transient_onsets(audio: torch.Tensor, sr: int):
    mono = audio.mean(dim=0).numpy()
    frame_size = int(sr * 0.01)  # 10ms frame
    energy = np.array(
        [
            np.sum(mono[i : i + frame_size] ** 2)
            for i in range(0, len(mono) - frame_size, frame_size)
        ]
    )
    min_separation_samples = int((MIN_SEPARATION_MS / 1000) * sr)
    onset_frames = np.where(energy > ENERGY_THRESHOLD)[0]

    transients = []
    if len(onset_frames) > 0:
        transients.append(onset_frames[0] * frame_size)

        for i in range(1, len(onset_frames)):
            current_sample = onset_frames[i] * frame_size
            last_hit_sample = transients[-1]

            if current_sample - last_hit_sample > min_separation_samples:
                transients.append(current_sample)

    return transients
