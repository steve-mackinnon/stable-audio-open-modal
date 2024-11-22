from math import floor
import argparse
import io
import numpy as np
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

from transient_detector import detect_transient_onsets


def generate_audio_sample(
    *,
    model: any,
    prompt: str,
    steps: int = 50,
    cfg_scale: int,
    length: float,
    trim_extra_hits: bool,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using device", device)
    sample_rate = 44100
    sample_size = floor(sample_rate * length)

    model = model.to(device)

    # Set up text and timing conditioning
    conditioning = [
        {
            "prompt": prompt + ". Generate a SINGLE DRUM HIT followed by silence.",
            "seconds_start": 0,
            "seconds_total": length,
        }
    ]

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
    if trim_extra_hits:
        transients = detect_transient_onsets(output, sample_rate)
        # The stable audio open model will sometimes generate multiple drum hits
        # despite requesting a single hit in the prompt. Below we use a basic transient
        # detector to trim extra hits. It's not perfect, but generally works pretty well.
        if len(transients) > 1:
            print(f"Trimming extra drum hits at {transients[1]}")
            output = output[:, : transients[1]]

    output = trim_trailing_silence(output, sample_rate)
    output = apply_fade_out(output, sample_rate).mul(32767).to(torch.int16).cpu()

    byte_io = io.BytesIO()
    torchaudio.save(byte_io, output, sample_rate, format="wav")
    byte_io.seek(0)  # Move to the beginning of the stream

    return byte_io


def trim_trailing_silence(audio: torch.Tensor, sr: int):
    silence_threshold = 0.01
    silent_samples = 0
    found_onset = False
    silence_after_index = None
    allowed_silent_samples = sr * 0.1
    for i in range(audio.shape[1]):
        # Check if all channels are below the threshold
        if torch.all(torch.abs(audio[:, i]) < silence_threshold):
            silent_samples += 1
            if found_onset and silent_samples > allowed_silent_samples:
                silence_after_index = i
                break
        else:
            silent_samples = 0
            found_onset = True
    if silence_after_index is None:
        return audio
    return audio[:, :silence_after_index]


def apply_fade_out(audio: torch.Tensor, sr: int, fade_out_ms: int = 10):
    fade_out_samples = int(fade_out_ms / 1000 * sr)
    fade_out_curve = torch.logspace(np.log10(1), np.log10(0.000001), fade_out_samples)
    audio[:, -fade_out_samples:] *= fade_out_curve
    return audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument(
        "--steps",
        type=int,
        required=False,
        default=50,
        help="The number of diffusion steps to run the model for. More steps generally result in higher quality audio at the cost of slower generation time.",
    )
    parser.add_argument(
        "--cfg_scale",
        type=int,
        required=False,
        default=7,
        help="Controls how closely the model matches the prompt. Recommended range is [7, 14].",
    )
    parser.add_argument(
        "--samples",
        type=int,
        required=False,
        default=1,
        help="The number of output samples to generate",
    )
    parser.add_argument(
        "--length",
        type=float,
        required=False,
        default=0.7,
        help="The max length of the generated audio in seconds",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default=".",
        help="The directory to save the output samples to",
    )
    parser.add_argument(
        "--skip_trim",
        required=False,
        action="store_true",
        default=False,
        help="Disable trimming of transient hits beyond the first one.",
    )
    args = parser.parse_args()

    # Download model
    model, _ = get_pretrained_model("stabilityai/stable-audio-open-1.0")

    for i in range(args.samples):
        output = generate_audio_sample(
            model=model,
            prompt=args.prompt,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            length=args.length,
            trim_extra_hits=not args.skip_trim,
        )
        with open(f"{args.output_dir}/output_{i}.wav", "wb") as f:
            f.write(output.getvalue())


if __name__ == "__main__":
    main()
