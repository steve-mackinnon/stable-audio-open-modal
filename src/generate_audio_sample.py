import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from math import floor
import argparse
import io

OUTPUT_LENGTH_SECONDS = 0.7


def generate_audio_sample(
    *, model: any, prompt: str, steps: int = 50, cfg_scale: int = 7
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using device", device)
    sample_rate = 44100
    sample_size = floor(sample_rate * OUTPUT_LENGTH_SECONDS)

    model = model.to(device)

    # Set up text and timing conditioning
    conditioning = [
        {
            "prompt": prompt + ". Generate a SINGLE DRUM HIT followed by silence.",
            "seconds_start": 0,
            "seconds_total": OUTPUT_LENGTH_SECONDS,
        }
    ]

    print("running model")
    # Generate stereo audio
    output = generate_diffusion_cond(
        model,
        steps=steps,
        cfg_scale=cfg_scale,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=device,
    )

    # Rearrange audio batch to a single sequence
    output = rearrange(output, "b d n -> d (b n)")

    # Peak normalize, clip, convert to int16, and save to file
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1)
    output = trim_trailing_silence(output, sample_rate).mul(32767).to(torch.int16).cpu()

    byte_io = io.BytesIO()
    torchaudio.save(byte_io, output, sample_rate, format="wav")
    byte_io.seek(0)  # Move to the beginning of the stream

    return byte_io


def trim_trailing_silence(audio: torch.Tensor, sr: int):
    SILENCE_THRESHOLD = 0.005
    silent_samples = 0
    found_onset = False
    silence_after_index = None
    allowed_silent_samples = sr * 0.1
    for i in range(audio.shape[1]):
        # Check if all channels are below the threshold
        if torch.all(torch.abs(audio[:, i]) < SILENCE_THRESHOLD):
            silent_samples += 1
            if found_onset and silent_samples > allowed_silent_samples:
                silence_after_index = i
                break
        else:
            silent_samples = 0
            found_onset = True
    if silence_after_index is None:
        print("No silence found after first sample")
        return audio
    print(f"Trimmed audio to {silence_after_index} samples")
    return audio[:, :silence_after_index]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--steps", type=int, required=False, default=50)
    parser.add_argument("--cfg_scale", type=int, required=False, default=7)
    parser.add_argument("--num_samples", type=int, required=False, default=1)
    args = parser.parse_args()

    # Download model
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    print("got model")

    for i in range(args.num_samples):
        bytes = generate_audio_sample(
            model=model, prompt=args.prompt, steps=args.steps, cfg_scale=args.cfg_scale
        )
        with open(f"output_{i}.wav", "wb") as f:
            f.write(bytes.getvalue())
