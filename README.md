## Environment setup

1. Install miniconda: https://docs.conda.io/en/latest/miniconda.html
2. Setup the conda environment

   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment

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

- `--steps`: Number of inference steps (default: 50)
- `--cfg_scale`: Classifier-free guidance scale (default: 7.0; recommended range 7 - 13)
- `--num_samples`: Number of audio samples to generate (default: 1)

## Running on Modal

To deploy the app to run inference on [Modal](https://modal.com), you'll need to:

1. Create a Modal account
2. Create a Huggingface account and API token.
3. Sign the agreement to use the [Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) model.
4. Setup secrets for the Modal app with the following environment variables:
   - `HF_TOKEN`: Your Huggingface API token
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
