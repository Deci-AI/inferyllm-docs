# InferyLLM
Inference engine and server for LLMs

## Requirements
1. Cuda driver 12.2 [download](https://developer.nvidia.com/cuda-downloads)
2. Python 3.11 [download](https://www.python.org/downloads/release/python-3110/)
3. Supported GPUs: NVIDIA A100, NVIDIA A10 <br>
    3.1: Memory: for optimal performance, should be at least 24G (preferably 32G)

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

docker pull docker pull deci.jfrog.io/deci-external-docker-local/infery-llm:0.0.1
```

By default, running the container will start a server on port 0.0.0.0:8080. The container simply enters the "local" 
entrypoint, thus you may pass the same exact arguments to it.

```bash
docker run --runtime=nvidia infery-llm:0.0.1 --model-name Deci/DeciLM-6b --port 9000
```

### Generation
Assuming you have a running server listening at `address:port`, you may submit generation requests to it like so:

```python
from infery_llm.client import LLMClient

client = LLMClient("http://address:port")

# submit a single prompt
result = client.generate("Write a short story about a dragon who was hungry:", max_new_tokens=10)
print(result.outputs[0])

# submit a batch of prompts
result = client.generate(["Write a short story about a dragon who was hungry:", "5 important facts about the prime minister of France are:"], max_new_tokens=10)
[print(output) for output in result.outputs]
```

You may also submit a request using curl:

``` shell
curl -X POST http://address:port/generate -H 'Content-Type: application/json' -d '{"prompts":["Write a short story about a dragon who was hungry:"],"max_new_tokens": 15}'
```
