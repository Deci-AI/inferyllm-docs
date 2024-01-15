# InferyLLM

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Serving](#serving)
4. [Generation](#generation)
5. [Advanced Usage](#advanced-usage)

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
   * [Llama 2 models](https://huggingface.co/docs/transformers/model_doc/llama2)
   * [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
   * All fine-tuned variants of the above.
   * More models coming soon... (Falcon, MPT, etc)

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
InferyLLM may be used with a lean (client-only) installation or a full (client+server) installation.

**Client Installation**
```bash
# Install InferyLLM (along with LLMClient)
pip install --extra-index-url=https://[ARTIFACTORY USER]:[ARTIFACTORY TOKEN]@deci.jfrog.io/artifactory/api/pypi/deciExternal/simple infery-llm
```
**Client & Server Installation**
```bash
# Install InferyLLM (along with LLMClient)
pip install --extra-index-url=https://[ARTIFACTORY USER]:[ARTIFACTORY TOKEN]@deci.jfrog.io/artifactory/api/pypi/deciExternal/simple infery-llm

# Install server requirements (you may export DECI_ARTIFACTORY_USER and DECI_ARTIFACTORY_TOKEN env vars instead of passing them)
infery-llm install -s server --user [ARTIFACTORY USER] --token [ARTIFACTORY TOKEN]
```

For a more thorough explanation, please refer to the [Advanced Usage](#advanced-usage) and check out the `install` CLI command.

### Pulling the InferyLLM container

To pull an InferyLLM container from Deci's container registry:

```bash
# Log in to Deci's container registry
docker login --username [ARTIFACTORY USER] --password [ARTIFACTORY TOKEN] deci.jfrog.io

# Pull the container. You may be specify a version instead of "latest" (e.g. 0.0.5)
docker pull deci.jfrog.io/deci-external-docker-local/infery-llm:latest
```

## Serving

There are two ways to serve an LLM with InferyLLM:
1. Through a local entrypoint
2. By using the InferyLLM container

By default, InferyLLM serves at `0.0.0.0:8080` this is configurable through passing the `--host` and `--port` flags.       

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

## Generation
Assuming you have a running server listening at `127.0.0.1:9000`, you may submit generation requests to it like so:
1. Through InferyLLM's `LLMClient`:
```python
import asyncio
from infery_llm.client import LLMClient, GenerationParams

client = LLMClient("http://127.0.0.1:9000")

# set generation params (max_new_tokens, temperature, etc...)
gen_params = GenerationParams(max_new_tokens=100, top_p=0.95, top_k=None, temperature=0.1)

# Submit a single prompt and query results (along with metadata in this case)
result = client.generate("Write a story about a red cat ", generation_params=gen_params, return_metadata=True)
print(f"Output: {result.output}.\nGenerated Tokens :{result.metadata[0]['generated_token_count']}")

# Submit a batch of prompts
prompts = ["A receipe for making spaghetti: ", "5 interesting facts about the President of France are: ", "Write a short story about a dog named Snoopy: "]
result = client.generate(prompts, generation_params=gen_params)
[print(f"Prompt: {output['prompt']}\nGeneration: {output['output']}") for output in result.outputs]

# Use stop tokens
gen_params = GenerationParams(stop_str_tokens=[1524], stop_strs=["add tomatoes"], skip_special_tokens=True)
result = client.generate("A receipe for making spaghetti: ", generation_params=gen_params)

# Stream results
for text in client.generate("Will the real Slim Shady please ", generation_params=gen_params, stream=True):
    print(text, end="")
    
# Async generation is also supported from within async code:
async def example():
    result = await client.generate_async("AsyncIO is fun because ", generation_params=gen_params)
    print(result.output)
asyncio.run(example())
```
2. Through a `curl` command (assuming you have [cURL](https://curl.se/) installed)
``` bash
curl -X POST http://0.0.0.0:9000/generate \
-H 'Content-Type: application/json' \
-d '{"prompts":["def factorial(n: int) -> int:"], "generation_params":{"max_new_tokens": 500, "temperature":0.5, "top_k":50, "top_p":0.8}, "stream":true}'
```

## Advanced Usage

### CLI

InferyLLM and its CLI are rapidly accumulating more features. For example, the `infery-llm` CLI already allows to `benchmark`
with numerous configurations, to `prepare` model artifacts before serving in order to [cut down loading time](#lowering-loading-time),
and more. To see the available features you may simply pass `--help` to the `infery-llm` CLI or any of its subcommands:

For container users:
```bash
# Query infery-llm CLI help menu
docker run --entrypoint infery-llm --runtime=nvidia deci.jfrog.io/deci-external-docker-local/infery-llm:latest --help

# Query the infery-llm CLI's `benchmark` subcommand help menu
docker run --entrypoint infery-llm --runtime=nvidia deci.jfrog.io/deci-external-docker-local/infery-llm:latest benchmark --help
```

For local installation users:
```bash
# Query infery-llm CLI help menu
infery-llm --help

# Query the infery-llm CLI's `benchmark` subcommand help menu
infery-llm benchmark --help
```

### Lowering loading time (the `prepare` command)

InferyLLM has its own internal format and per-model artifact requirements. While the required artifacts are automatically
generated by the `infery-llm serve` command, you can also generate them ahead of time with the `infery-llm prepare`
command, thus drastically cutting down server start-time.

For container users:
```bash
# Create artifacts for serving a Deci/DeciCoder-1b and place the result in ~ on the host machine
docker run --rm --entrypoint infery-llm -v ~/:/models --runtime=nvidia deci.jfrog.io/deci-external-docker-local/infery-llm:latest prepare --hf-model Deci/DeciCoder-1b --output-dir /models/infery_llm_model

# Now serve the created artifact (specifically here on port 9000)
docker run --runtime=nvidia -p 9000:9000 -v ~/:/models deci.jfrog.io/deci-external-docker-local/infery-llm:latest --infery-model-dir /models/infery_llm_model --port 9000
```

For local installation users:
```bash
# Create artifacts for serving a Deci/DeciCoder-1b and place the result in ~ on the host machine
infery-llm prepare --hf-model Deci/DeciCoder-1b --output-dir /models/infery_llm_model

# Now serve the created artifact (specifically here on port 9000)
infery-llm serve --infery-model-dir /models/infery_llm_model --port 9000
```

**Important note on caching:** Just like 🤗 caches downloaded model weights in `~/.cache/huggingface`, InferyLLM 
caches the mentioned artifacts in `~/.cache/deci`. It does so for every model served.
This means that relaunching a local or containerized server (if it is the same container, not just the same image) will
automatically lower the loading time.

