# InferyLLM
An inference engine and server for LLMs by Deci.<br>
With Infery-LLM, you can supercharge the performance of your LLMs, boosting speed by up to 5x while maintaining the same accuracy.  
Unprecedented inference efficiency emerges when combining Deci’s open-source models such as DeciCoder or DeciLM 6B and Infery-LLM. Furthermore, Infery-LLM allows you to run larger models on more widely available and cost-effective GPUs by supporting different parallelism paradigms.<br>
Check out this [blog post](https://deci.ai/blog/decilm-15-times-faster-than-llama2-nas-generated-llm-with-variable-gqa/) for more details.

## Requirements
1. [CUDA 12.2](https://developer.nvidia.com/cuda-downloads)
2. [nvidia-docker-runtime](https://developer.nvidia.com/nvidia-container-runtime)
3. [Python 3.11](https://www.python.org/downloads/release/python-3110/)
4. Supported Models:
   * [DeciLM 6B](https://huggingface.co/Deci/DeciLM-6b)
   * [DeciLM 6B instruct](https://huggingface.co/Deci/DeciLM-6b-instruct)
   * [DeciCoder 1B](https://huggingface.co/Deci/DeciCoder-1b)
   * Coming soon - LLaMA, Falcon, Mistral, MPT
6. Supported GPUs: Compute capability >= 8.0 (e.g. A100, A10, L4, ...)<br>
   * Memory requirements depends on the model size.
        * DeciLM-6B - at least 24G. 
        * DeciCoder-1B - 16G is more than enough.
    
### New features:
- Support for sampling params ✅
- Support for stop string and stop tokens list ✅
- Support for FP16 ✅

### Backlog:
- Streaming tokens
- Advanced sampling techniques (beam search and others)
- Support for more models (MPT, Mistral, Falcon)
- More performance optimization and tuning

  
## Installation
To install the InferyLLM package you must get artifactory credentials from Deci:
### Client
``` shell
pip install --extra-index-url=https://[USERNAME]:[TOKEN]@deci.jfrog.io/artifactory/api/pypi/deciExternal/simple infery-llm[client]
```
### Server
``` shell
pip install --extra-index-url=https://[USERNAME]:[TOKEN]@deci.jfrog.io/artifactory/api/pypi/deciExternal/simple infery-llm[server]
```
### Local
``` shell
pip install --extra-index-url=https://[USERNAME]:[TOKEN]@deci.jfrog.io/artifactory/api/pypi/deciExternal/simple infery-llm[local]
```


## Usage
### Local server
You may run an inference server that serves a single LLM model with the `infery-llm-serve` cli tool. 
Use the `-h` flag to see all configurable options. An example of serving a `DeciLM-6b` model on port 9000:

```bash
infery-llm-serve --model-name Deci/DeciLM-6b --port 9000
```

### Container server
To pull the InferyLLM docker you must use the same artifactory credentials from above:
```bash
docker login --username [USERNAME] --password [PASSWORD] deci.jfrog.io

docker pull deci.jfrog.io/deci-external-docker-local/infery-llm:0.0.2
```

By default, running the container will start a server on port 0.0.0.0:8080. The container simply enters the "local" 
entrypoint, thus you may pass the same exact arguments to it.

```bash
docker run --runtime=nvidia infery-llm:0.0.2 --model-name Deci/DeciLM-6b --port 9000
```

### Generation
Assuming you have a running server listening at `127.0.0.1:9000`, you may submit generation requests to it like so:

```python
from infery_llm.client import LLMClient

client = LLMClient("http://127.0.0.1:9000")

# set generation params (max_new_tokens, temperature, etc...)
gen_params = GenerationParams(max_new_tokens=100, top_p=0.95, top_k=0, temperature=0.1, do_sample=True)

# submit a single prompt and query results
result = client.generate("A receipe for making spaghetti: ", generation_params=gen_params)
print(result.outputs[0])

# submit a batch of prompts
prompts = ["A receipe for making spaghetti: ", "5 interesting facts about the President of France are: ", "Write a short story about a dog named Snoopy: "]
result = client.generate(prompts, generation_params=gen_params)
[print(output) for output in result.outputs]

# use stop tokens
gen_params = GenerationParams(do_sample=False, stop_str_tokens=[1524], stop_strs=["add tomatoes"], skip_special_tokens=True)
result = client.generate("A receipe for making spaghetti: ", generation_params=gen_params)
```

You may also submit a request using curl:

``` shell
curl -X POST http://address:port/generate -H 'Content-Type: application/json' -d '{"prompts":["Write a short story about a dragon who was hungry:"], "max_new_tokens": 15}'
```
