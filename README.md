## Development Environment Setup

The following are required:
- Python: >=3.12
- uv: install uv for package/project management, https://docs.astral.sh/uv/guides/install-python/
- git: https://github.com/git-guides/install-git


## Development

```shell
cd d-vllm
uv sync --all-extras --group dev
source .venv/bin/activate
```

#Model Download
To download the model weights manually, use the following command:
```shell
python download_model.py 
```

#Benchmark Testing 
It’s just about making random token sequences to test how well the system performs.
```shell

```
