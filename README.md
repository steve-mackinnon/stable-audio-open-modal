# Stable Audio Open Modal

This repo includes python code for running inference with the [Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) model. This can be run locally or hosted on [Modal](https://modal.com).

`generate_audio_sample.py` tweaks the provided prompt to attempt to generate a single "oneshot" sample like a drum hit. It then applies some post processing to the model output to trim extra hits and fade out the audio smoothly.

## Hugging Face setup

In order to access the Stable Audio Open model, you'll need to:
1. Create a [Hugging Face account](https://huggingface.co/)
2. Navigate to the [Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) model page and opt-in to gain access to the model
3. Create a [Hugging Face access token](https://huggingface.co/settings/tokens/new?tokenType=read) with read access
4. Copy the token and add it to your local env using the name HF_TOKEN:

For zsh, add this to your `~/.zshrc`:
```bash
export HF_TOKEN=myhftoken 
```

For fish, add this to your fish config (e.g. `~/.config/fish/config.fish`):
```bash
set -Ux HF_TOKEN myhftoken
```

## Local environment setup

1. Install miniconda: https://docs.conda.io/en/latest/miniconda.html
2. Setup the conda environment

   ```bash
   conda env create -f environment.yml
   ```

3. Activate it 

   ```bash
   conda activate stable-audio-open-modal
   ```

## Running locally

To run inference locally, you can run `generate_audio.py` after activating the conda environment.

For example:

```bash
python generate_audio.py --prompt "Massive metalic techno kick drum"
```

This will generate a file called `output_0.wav` in the current directory.

You can optionally provide the following arguments:

- `--steps`: Number of inference steps (default: 100)
- `--cfg_scale`: Classifier-free guidance scale (default: 7.0; recommended range 7 - 13)
- `--num_samples`: Number of audio samples to generate (default: 1)

## Running on Modal

To deploy the app to run inference on [Modal](https://modal.com), you'll need to:

1. Create a Modal account
2. Create a Hugging Face account and API token.
3. Sign the agreement to use the [Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) model.
4. Setup secrets for the Modal app with the following environment variables:
   - `HF_TOKEN`: Your Hugging Face API token
   - `AUTH_TOKEN`: A Bearer auth token you create to authenticate requests to the Modal app
5. Deploy the app with the following command:

   ```bash
   modal deploy src/api.py
   ```

Note, you can test the endpoint prior to deploying with the following command:

```bash
modal serve src/api.py
```

And hit the endpoint with a POST request locally. This assumes you have set the `AUTH_TOKEN` environment variable.

```bash
curl -X POST https://your-modal-endpoint.modal.run \
  -H "Authorization: Bearer $SAMPLE_GEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Dub techno snare"
  }' --output "modal-out.wav"
```
