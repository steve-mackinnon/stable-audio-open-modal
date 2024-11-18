## Environment setup

```bash
conda env create -f environment.yml
```

Note: I had to `brew install libsndfile` then copy it into the conda environment to get this to work.

```bash
cp /opt/homebrew/lib/libsndfile.dylib /opt/miniconda3/envs/sample-gen-api/lib/python3.11/site-packages/_soundfile_data/libsndfile.dylib
```
