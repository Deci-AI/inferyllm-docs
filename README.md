# InferyLLM

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Serving](#serving)
4. [Generation](#generation)
5. [Advanced Usage](#advanced-usage)
6. [Backlog](#backlog)

## Introduction
InferyLLM is a high-performance engine and server for running LLM inference.

### InferyLLM is fast
- Optimized CUDA kernels for MQA and GQA
- Continuous batching using a paged KV cache and custom paged attention kernels 
- Kernel autotuning capabilities with automatic selection of the optimal kernels and parameters on the given HW
- Support for extremely efficient LLMs, designed to reach SOTA throughput

### InferyLLM is easy to use
- Containerized OR local entrypoint servers
- Simple, minimal-dependency python client
- Seamless integration with 🤗 model hub

### Model support
   * [DeciLM 6B](https://huggingface.co/Deci/DeciLM-6b)
   * [DeciLM 6B instruct](https://huggingface.co/Deci/DeciLM-6b-instruct)
   * [DeciCoder 1B](https://huggingface.co/Deci/DeciCoder-1b)
   * Fine-tuned variants of the above.
   * Other models coming soon... (LLaMA variants, Falcon, Mistral, MPT)

### Supported GPUs
* Compute capability >= 8.0 (e.g. A100, A10, L4, ...)<br>
* Memory requirements depends on the model size.
* DeciLM-6B - at least 24G.
* DeciCoder-1B - 16G is more than enough.
  
## Installation
### Prerequisites
Before you begin, verify you have an artifactory **user** and **token** from Deci.<br> 
Then, ensure you have met the following system requirements:

- General requirements:
  - Python >= 3.11
  - [CUDA ToolKit >= 12.1](https://developer.nvidia.com/cuda-downloads)
- For local serving:
  - GLIBC >= 2.31
  - GCC, G++ >= 11.3
  - `gcc`, `g++` and `nvcc` in your `$PATH` at the time of installation
- For containerized serving:
  - [nvidia-container-runtime >= 1.13.4](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/release-notes.html)

### Installing locally
Using the InferyLLM `LLMClient` requires a very lean installation. For serving locally, an extra installation step is required.

```bash
# Install InferyLLM (along with LLMClient)
pip install --extra-index-url=https://[ARTIFACTORY USER]:[ARTIFACTORY TOKEN]@deci.jfrog.io/artifactory/api/pypi/deciExternal/simple infery-llm

# Now, for local serving install the additional requirements (you may export DECI_ARTIFACTORY_USER and DECI_ARTIFACTORY_TOKEN env vars instead of passing them)
infery-llm install --subpackage server --user [ARTIFACTORY USER] --token [ARTIFACTORY TOKEN]
```

For a more thorough explanation, please refer to the [Advanced Usage](#advanced-usage)

### Pulling the InferyLLM container

To pull an InferyLLM container from Deci's container registry:

```bash
# Log in to Deci's container registry
docker login --username [ARTIFACTORY USER] --password [ARTIFACTORY TOKEN] deci.jfrog.io

# Pull the container. [VERSION TAG] may be a specific version (e.g. 0.0.3) or "latest"
docker pull deci.jfrog.io/deci-external-docker-local/infery-llm:[VERSION TAG]
```

## Serving

There are two ways to serve an LLM with InferyLLM:
1. Through a local entrypoint
2. By using the InferyLLM container

**It is highly advised to serve using the container rather than the local entrypoint**

### Serving with a container

Assuming you have pulled the container as shown in the [Installation](#pulling-the-inferyllm-container) section,
running the server is a simple one-liner. You can also use the container to query the serving CLI `help` for all 
available serving flags and defaults:

```bash
# Serve Deci/DeciLM-6b (from HF hub) on port 9000
docker run --runtime=nvidia -p 9000:9000 deci.jfrog.io/deci-external-docker-local/infery-llm:[VERSION TAG] --model-name Deci/DeciLM-6b --port 9000

# See all serving CLI options and defaults
docker run --rm --runtime=nvidia deci.jfrog.io/deci-external-docker-local/infery-llm:[VERSION TAG] --help
```

Notice that a HuggingFace token may be passed as an environment variable (using the docker `-e` flag) or as a CLI parameter

### Serving with a local entry point

Assuming you have installed the `infery-llm` local serving requirements, you may use the InferyLLM CLI as a server entrypoint:
```bash
# Serve Deci/DeciLM-6b (from HF hub) on port 9000
infery-llm serve --model-name Deci/DeciLM-6b --port 9000

# See all serving options
infery-llm serve --help
```

### Lowering loading time

Loading a HuggingFace-format LLM with InferyLLM requires the following procedure:
1. Downloading the model configuration
2. Downloading the model weights
3. Downloading the required tokenizer
4. Converting the model weights to a format optimized for InferyLLM's kernels
5. Autotuning the converted model

While this flow is fully automated and encapsulated within the `infery-llm serve` command, you can perform the entire flow
ahead of time with the `infery-llm prepare` command and pass the created artifact straight to the serving entry point.

For container users:
```bash
# Create artifacts for serving a Deci/DeciCoder-1b and place the result in ~ on the host machine
docker run --rm --entrypoint infery-llm -v ~/:/models --runtime=nvidia deci.jfrog.io/deci-external-docker-local/infery-llm:[VERSION TAG] prepare --hf-model Deci/DeciCoder-1b --output-dir /models/infery_llm_model

# Now serve the created artifact
docker run --runtime=nvidia -v ~/:/models deci.jfrog.io/deci-external-docker-local/infery-llm:[VERSION TAG] --infery-model-dir /models/infery_llm_model
```

For local serving users:
```bash
# Create artifacts for serving a Deci/DeciCoder-1b and place the result in ~ on the host machine
infery-llm prepare --hf-model Deci/DeciCoder-1b --output-dir /models/infery_llm_model

# Now serve the created artifact
infery-llm serve --infery-model-dir /models/infery_llm_model
```

**Important note on caching:** Just like HuggingFace caches downloaded model weights in `~/.cache/huggingface`, InferyLLM 
caches the `prepare` artifacts in `~/.cache/deci`. It does so for every model served.
This means that relaunching a local or containerized server (if it is the same container, not just the same image) will
automatically lower the loading time.

## Generation
Assuming you have a running server listening at `127.0.0.1:9000`, you may submit generation requests to it like so:
1. Through InferyLLM's `LLMClient`:
```python
from infery_llm.client import LLMClient, GenerationParams

client = LLMClient("http://127.0.0.1:9000")

# set generation params (max_new_tokens, temperature, etc...)
gen_params = GenerationParams(max_new_tokens=100, top_p=0.95, top_k=0, temperature=0.1)

# submit a single prompt and query results
result = client.generate("A receipe for making spaghetti: ", generation_params=gen_params)
print(result.outputs[0])

# submit a batch of prompts
prompts = ["A receipe for making spaghetti: ", "5 interesting facts about the President of France are: ", "Write a short story about a dog named Snoopy: "]
result = client.generate(prompts, generation_params=gen_params)
[print(output) for output in result.outputs]

# use stop tokens
gen_params = GenerationParams(stop_str_tokens=[1524], stop_strs=["add tomatoes"], skip_special_tokens=True)
result = client.generate("A receipe for making spaghetti: ", generation_params=gen_params)
```
2. Through a `curl` command (assuming you have [cURL](https://curl.se/) installed)
``` bash
curl -X POST http://127.0.0.1:8080/generate -H 'Content-Type: application/json' \
-d '{"prompts":["def factorial(n: int) -> int:"], "generation_params":{"max_new_tokens": 500, "temperature":0.5, "top_k":50, "top_p":0.8}}'
```

## Advanced Usage

InferyLLM and its CLI are rapidly accumulating more features. For example, the `infery-llm` CLI already allows to `benchmark`
with numerous configurations, to `prepare` model artifacts before serving in order to [cut down loading time](#lowering-loading-time),
and more. To see the available features you may simply pass `--help` to the `infery-llm` CLI or any of its subcommands:

For container users:
```bash
# Query infery-llm CLI help menu
docker run --entrypoint infery-llm --runtime=nvidia deci.jfrog.io/deci-external-docker-local/infery-llm:[VERSION TAG] --help

# Query the infery-llm CLI's `benchmark` subcommand help menu
docker run --entrypoint infery-llm --runtime=nvidia deci.jfrog.io/deci-external-docker-local/infery-llm:[VERSION TAG] benchmark --help
```

For local serving users:
```bash
# Query infery-llm CLI help menu
infery-llm --help

# Query the infery-llm CLI's `benchmark` subcommand help menu
infery-llm benchmark --help
```

## Backlog

- Token streaming
- Advanced sampling techniques (beam search and others)
- Support for more models (MPT, Mistral, Falcon)
- Further performance optimization and tuning
