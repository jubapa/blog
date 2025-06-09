---
title: Quantization
author: jubapa
date: 2025-06-07 00:10:00 +0800
categories: [Blogging, AI, ollama, vllm]
tags: [AI]
---

## Starting point

Everything started as I did a test with ollama and the [qwen3:32b](https://ollama.com/library/qwen3:32b) model, which it worked and I did want to test it with vllm. To my surprise there was an awfull error due to the RAM was not enough.

```bash
# Error trying to run the Qwen/Qwen3-32B model in vllm :(
[vllm]                 | INFO 05-29 04:44:25 [gpu_model_runner.py:1276] Starting to load model Qwen/Qwen3-32B...
[vllm]                 | ERROR 05-29 04:44:26 [core.py:387] EngineCore hit an exception: Traceback (most recent call last):
.
.
[vllm]                 | ERROR 05-29 04:44:26 [core.py:387] torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 100.00 MiB. GPU 0 has a total capacity of 23.64 GiB of which 58.81 MiB is free. Process 1293514 has 22.82 GiB memory in use. Of the allocated memory 22.37 GiB is allocated by PyTorch, and 18.93 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

So I checked the [Qwen3 ollama library](https://ollama.com/library/qwen3:32b). I realized, in the parameters of the model these parameters can be seen:



| Paratemer    | Value  |
| ------------ | ------ |
| Arch         | Qwen3  |
| Parameters   | 32B    |
| Quantization | Q4_K_M |

```bash
# Checking the size of the Qwen3 model in ollama
(.venv) jubapa@FedoraAI:~$ ollama ls
NAME                  ID              SIZE      MODIFIED           
qwen3:32b             e1c9f234c6eb    20 GB     4 weeks ago
```

So I started to dig a little bit more about Quantization

### What is Quantization and its purpose ?

LLM needs quite a lot of memory and gpu power. The idea is to reduce the ammount of memory needed and gpu power. This reduction has some cost. The cost is the preccision of the model, so the result model will be less accurate. A balance between less resource needed and lost accuracy needs to be found.

Benefits:
 - Reduce memory footprint
 - Faster inference
 - Energy efficiency
 - Reduce the cost of the hardware needed

Disadvantages:
 - Decrease accuracy

A little deeper explanation about [quantization](https://huggingface.co/docs/optimum/concept_guides/quantization)


### Distillation

Distillation is use a big LLM to train a small LLM, to try to obtain the same results.
So there is a role teacher/student between the LLMs and the distillation method requires
to train the student LLM (smaller one)

Benefits
- The output LLM can be configured ad-hoc 
- If the learning process was completed the LLMs can have very similar responses in a smaller LLM

Disadvantages:
- Requires training
    - Energy
    - Time
    - Having a previous model
    
### How to do it locally

#### Ollama


If you check the [ollama qwen3 tags](https://ollama.com/library/qwen3/tags) there is a table with the models size context and input

| Model            | Size | Context | Input |
| ---------------- | ---- | ------- | ----- |
| qwen3:32b-q4_K_M | 20GB | 40K     | Text  |
| qwen3:32b-fp16   | 66GB | 40K     | Text  |
| qwen3:32b        | 20GB | 40K     | Text  | 

There is a hash 030ee887880f that is the same for qwen3:32b and qwen3:32b-q4_K_M so they are the same.

#####  Crete a q4_K_M from the fp16 bigger model

```bash
# File with Model to quantize for ollama
(.venv) jubapa@FedoraAI:~$ cat modelfile 
FROM qwen3:32b-fp16
```

```bash
# Quantize the model to Q4_K_M
(.venv) jubapa@FedoraAI:~$ time ollama create jjb -q Q4_K_M -f modelfile 
gathering model components 
pulling manifest 
pulling 50c6fb7ac4eb... 100% ▕██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏  65 GB                         
pulling ae370d884f10... 100% ▕██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏ 1.7 KB                         
pulling d18a5cc71b84... 100% ▕██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏  11 KB                         
pulling cff3f395ef37... 100% ▕██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏  120 B                         
pulling 5b29bbb8b253... 100% ▕██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏  485 B                         
verifying sha256 digest 
writing manifest 
success 
quantizing F16 model to Q4_K_M 
creating new layer sha256:0bad24949533f1ecf70b82605f5bc754af0dfe7d0c922b92421877bf26e32557 
using existing layer sha256:ae370d884f108d16e7cc8fd5259ebc5773a0afa6e078b11f4ed7e39a27e0dfc4 
using existing layer sha256:d18a5cc71b84bc4af394a31116bd3932b42241de70c77d2b76d69a314ec8aa12 
using existing layer sha256:cff3f395ef3756ab63e58b0ad1b32bb6f802905cae1472e6a12034e4246fbbdb 
writing manifest 
success 

real	16m38.950s
user	0m1.335s
sys	0m1.061s
# It took less than 17 minutes to quantize
```


```bash
# Checking the models in ollama and its size
jubapa@FedoraAI:~/repos/openwebui$ ollama ls 
NAME                  ID              SIZE      MODIFIED          
jjb:latest            006eaf53a505    20 GB     About an hour ago    
qwen3:32b-fp16        00dc0cb60a08    65 GB     About an hour ago    
qwen3:32b             e1c9f234c6eb    20 GB     4 weeks ago      
```

[Good table about size/accuracy](https://github.com/ggml-org/llama.cpp/discussions/2094#discussioncomment-6351796)

#### vllm

I tried to use the [llm-copressor](https://github.com/vllm-project/llm-compressor/) to quantize the model but with not succeed, probably because I am still learning about all this.
There is a quite interesting article about it at [developers.redhat](https://developers.redhat.com/articles/2024/08/14/llm-compressor-here-faster-inference-vllm)

So I tried different approach, checking on the things that have done by others at [Quantization](https://huggingface.co/models?other=base_model:quantized:Qwen/Qwen3-32B) filter by [RedHat](https://huggingface.co/models?other=base_model:quantized:Qwen%2FQwen3-32B&sort=trending&search=RedHat).
There are two models:
- https://huggingface.co/RedHatAI/Qwen3-32B-quantized.w4a16 
- https://huggingface.co/RedHatAI/Qwen3-32B-FP8-dynamic


Checking the first one, [RedHatAI/Qwen3-32B-quantized.w4a16](https://huggingface.co/RedHatAI/Qwen3-32B-quantized.w4a16) from [Optimizations](https://huggingface.co/RedHatAI/Qwen3-32B-quantized.w4a16#model-optimizations) "reducing the disk size and GPU memory requirements by approximately 75%." and there is also a [creation](https://huggingface.co/RedHatAI/Qwen3-32B-quantized.w4a16#creation) section on which explains how this model was created (Click on cration details):


Create the environment to play, check [pytotch]({% post_url 2025-06-06-pytorch %}) in case of issues 
```bash
$ # Create a virtual environment
$ virtualenv .test
$ # Active the test environment
$ source .test/bin/activate
$ # install llmcompressor to quantize the model
(.test) $ pip install llcompressor
.
.
Using cached yarl-1.20.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (349 kB)
Installing collected packages: pytz, nvidia-ml-py, nvidia-cusparselt-cu12, mpmath, xxhash, urllib3, tzdata, typing-extensions, tqdm, sympy, six, setuptools, safetensors, regex, pyyaml, pynvml, pyarrow, psutil, propcache, pillow, packaging, nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufile-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, numpy, networkx, multidict, MarkupSafe, loguru, idna, hf-xet, fsspec, frozenlist, filelock, dill, charset-normalizer, certifi, attrs, annotated-types, aiohappyeyeballs, yarl, typing-inspection, triton, requests, python-dateutil, pydantic-core, nvidia-cusparse-cu12, nvidia-cufft-cu12, nvidia-cudnn-cu12, multiprocess, jinja2, aiosignal, pydantic, pandas, nvidia-cusolver-cu12, huggingface-hub, aiohttp, torch, tokenizers, transformers, datasets, accelerate, compressed-tensors, llmcompressor
Successfully installed MarkupSafe-3.0.2 accelerate-1.7.0 aiohappyeyeballs-2.6.1 aiohttp-3.12.11 aiosignal-1.3.2 annotated-types-0.7.0 attrs-25.3.0 certifi-2025.4.26 charset-normalizer-3.4.2 compressed-tensors-0.9.4 datasets-3.6.0 dill-0.3.8 filelock-3.18.0 frozenlist-1.6.2 fsspec-2025.3.0 hf-xet-1.1.3 huggingface-hub-0.32.4 idna-3.10 jinja2-3.1.6 llmcompressor-0.5.1 loguru-0.7.3 mpmath-1.3.0 multidict-6.4.4 multiprocess-0.70.16 networkx-3.5 numpy-1.26.4 nvidia-cublas-cu12-12.6.4.1 nvidia-cuda-cupti-cu12-12.6.80 nvidia-cuda-nvrtc-cu12-12.6.77 nvidia-cuda-runtime-cu12-12.6.77 nvidia-cudnn-cu12-9.5.1.17 nvidia-cufft-cu12-11.3.0.4 nvidia-cufile-cu12-1.11.1.6 nvidia-curand-cu12-10.3.7.77 nvidia-cusolver-cu12-11.7.1.2 nvidia-cusparse-cu12-12.5.4.2 nvidia-cusparselt-cu12-0.6.3 nvidia-ml-py-12.575.51 nvidia-nccl-cu12-2.26.2 nvidia-nvjitlink-cu12-12.6.85 nvidia-nvtx-cu12-12.6.77 packaging-25.0 pandas-2.3.0 pillow-11.2.1 propcache-0.3.1 psutil-7.0.0 pyarrow-20.0.0 pydantic-2.11.5 pydantic-core-2.33.2 pynvml-12.0.0 python-dateutil-2.9.0.post0 pytz-2025.2 pyyaml-6.0.2 regex-2024.11.6 requests-2.32.3 safetensors-0.5.3 setuptools-80.9.0 six-1.17.0 sympy-1.14.0 tokenizers-0.21.1 torch-2.7.1 tqdm-4.67.1 transformers-4.52.4 triton-3.3.1 typing-extensions-4.14.0 typing-inspection-0.4.1 tzdata-2025.2 urllib3-2.4.0 xxhash-3.5.0 yarl-1.20.0


[notice] A new release of pip is available: 23.3.2 -> 25.1.1
[notice] To update, run: pip install --upgrade pip
```


Creation of the files...and play with it.
```bash
# Copy the file from Qwen3-32B-quantized.w4a16#creation and mofidied a little bit 
$ cat qwen-redhat.py
from llmcompressor.modifiers.quantization import GPTQModifier
#from llmcompressor.transformers import oneshot
from llmcompressor import oneshot
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Load model
model_stub = "Qwen/Qwen3-32B"
model_name = model_stub.split("/")[-1]

num_samples = 1024
max_seq_len = 8192

#model = AutoModelForCausalLM.from_pretrained(model_stub)
# Offload to CPU in case does not fit
model = AutoModelForCausalLM.from_pretrained(model_stub, device_map="auto", torch_dtype="auto", trust_remote_code=True )

tokenizer = AutoTokenizer.from_pretrained(model_stub)

def preprocess_fn(example):
  return {"text": tokenizer.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False)}

ds = load_dataset("neuralmagic/LLM_compression_calibration", split="train")
ds = ds.map(preprocess_fn)

# Configure the quantization algorithm and scheme
recipe = GPTQModifier(
    ignore=["lm_head"],
    sequential_targets=["Qwen3DecoderLayer"],
    targets="Linear",
    scheme="W4A16",
    dampening_frac=0.1,
)

# Apply quantization
oneshot(
    model=model,
    dataset=ds, 
    recipe=recipe,
    max_seq_length=max_seq_len,
    num_calibration_samples=num_samples,
)

# Save to disk in compressed-tensors format
save_path = model_name + "-quantized.w4a16"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model and tokenizer saved to: {save_path}")
```


Unfortunatelly my server is not big enough

```bash
(.venv) jubapa@FedoraAI:~$ # Try to quantize
(.venv) jubapa@FedoraAI:~/repos/llm-compressor/examples/big_models_with_accelerate$ python qwen-redhat.py 
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 17/17 [00:12<00:00,  1.37it/s]
Some parameters are on the meta device because they were offloaded to the cpu.
Tokenizing: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:27<00:00, 360.64 examples/s]
2025-06-02T13:50:49.735903+0200 | reset | INFO - Compression lifecycle reset
2025-06-02T13:50:49.744700+0200 | from_modifiers | INFO - Creating recipe from modifiers
Preparing intermediates cache: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [00:07<00:00, 128.38it/s]
(1/65): Calibrating:   0%|                                                                                                                                                                                                                                                                           | 0/1024 [00:00<?, ?it/s]
/home/jubapa/repos/llm-compressor/.venv/lib/python3.12/site-packages/llmcompressor/modifiers/quantization/gptq/base.py:202: UserWarning: Falling back to layer_sequential pipeline
  warnings.warn("Falling back to layer_sequential pipeline")
Preparing intermediates cache:  58%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                                         | 589/1024 [01:17<01:30,  4.79it/s]
Killed
```

Checking the use of the gpu with nvtop
```bash
Device 0 [NVIDIA GeForce RTX 4090] PCIe GEN 3@ 8x RX: 350.0 KiB/s TX: 2.190 GiB/s
GPU 2520MHz MEM 10251MH TEMP  50°C  FAN  36%   POW  53 / 450 W
GPU[|||||||||||                     32%] MEM[||||||||||||||||||22.746Gi/23.988Gi]
   ┌────────────────────────────────────────────────────────────────────────────┐
100│GPU0 %                                                                      │
   │GPU0 mem%                                        ┌──────────────────────────│
   │                                 ┌───────────────┘                          │
   │─────────────────────────────────┘                                          │
 75│                                                      			│
   │                                                      			│
   │                                                      			│
   │                                                      			│
 50│                                                      			│
   │                                                      			│
   │                                                ┌───────────────────────────│
   │                                                │                           │
 25│                                                │                           │
   │                                ┌─┐             │                           │
   │                                │ │             │                           │
  0│────────────────────────────────┘ └─────────────┘                           │
   └38s───────────────28s────────────────19s────────────────9s────────────────0s┘
    PID   USER DEV     TYPE  GPU        GPU MEM    CPU  HOST MEM Command         
 776194 jubapa   0  Compute  26%  22438MiB  91%   100%  28975MiB python qwen-redh
   6099 jubapa   0  Graphic   0%    216MiB   1%     0%    376MiB /usr/lib64/firef
   2346 jubapa   0  Graphic   5%    116MiB   0%     3%    221MiB /usr/bin/gnome-s
 837875 jubapa   0  Graphic   0%     64MiB   0%     0%     27MiB /usr/bin/gnome-t
1177941 jubapa   0  Graphic   0%     61MiB   0%     0%     28MiB /usr/bin/nautilu
F2Setup   F6Sort    F9Kill    F10Quit    F12Save Config                          

```

Checking the use of the server ram 
```bash
jubapa@FedoraAI:~/repos/openwebui$ free -h
               total        used        free      shared  buff/cache   available
Mem:           125Gi        86Gi        39Gi        94Mi       1.4Gi        39Gi
Swap:          8.0Gi       3.7Gi       4.3Gi
jubapa@FedoraAI:~/repos/openwebui$ free -h
               total        used        free      shared  buff/cache   available
Mem:           125Gi       118Gi       883Mi        98Mi       7.5Gi       7.2Gi
Swap:          8.0Gi       3.1Gi       4.9Gi
jubapa@FedoraAI:~/repos/openwebui$ free -h
               total        used        free      shared  buff/cache   available
Mem:           125Gi       120Gi       854Mi        98Mi       5.9Gi       5.5Gi
Swap:          8.0Gi       3.1Gi       4.9Gi
jubapa@FedoraAI:~/repos/openwebui$ free -h
               total        used        free      shared  buff/cache   available
Mem:           125Gi       122Gi       844Mi        98Mi       3.9Gi       3.5Gi
Swap:          8.0Gi       3.1Gi       4.9Gi
jubapa@FedoraAI:~/repos/openwebui$ free -h
               total        used        free      shared  buff/cache   available
Mem:           125Gi       124Gi       784Mi        98Mi       2.0Gi       1.6Gi
Swap:          8.0Gi       3.1Gi       4.9Gi
jubapa@FedoraAI:~/repos/openwebui$ free -h
               total        used        free      shared  buff/cache   available
Mem:           125Gi       117Gi       8.5Gi        89Mi       780Mi       8.0Gi
Swap:          8.0Gi       4.7Gi       3.3Gi
jubapa@FedoraAI:~/repos/openwebui$ free -h
               total        used        free      shared  buff/cache   available
Mem:           125Gi       3.8Gi       122Gi       7.4Mi       724Mi       121Gi
Swap:          8.0Gi       4.1Gi       3.9Gi
```

Using the RedHatAI/Qwen3-32B-quantized.w4a16 model didnt work either, due to the big context. The following error was shown:

```bash
[vllm]                 | INFO 06-02 05:16:16 [backends.py:144] Compiling a graph for general shape takes 59.77 s
[vllm]                 | INFO 06-02 05:17:18 [monitor.py:33] torch.compile takes 76.34 s in total
[vllm]                 | ERROR 06-02 05:17:18 [core.py:387] EngineCore hit an exception: Traceback (most recent call last):
[vllm]                 | ERROR 06-02 05:17:18 [core.py:387]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 378, in run_engine_core
[vllm]                 | ERROR 06-02 05:17:18 [core.py:387]     engine_core = EngineCoreProc(*args, **kwargs)
[vllm]                 | ERROR 06-02 05:17:18 [core.py:387]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[vllm]                 | ERROR 06-02 05:17:18 [core.py:387]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 320, in __init__
[vllm]                 | ERROR 06-02 05:17:18 [core.py:387]     super().__init__(vllm_config, executor_class, log_stats)
[vllm]                 | ERROR 06-02 05:17:18 [core.py:387]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 71, in __init__
[vllm]                 | ERROR 06-02 05:17:18 [core.py:387]     self._initialize_kv_caches(vllm_config)
[vllm]                 | ERROR 06-02 05:17:18 [core.py:387]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 138, in _initialize_kv_caches
[vllm]                 | ERROR 06-02 05:17:18 [core.py:387]     get_kv_cache_config(vllm_config, kv_cache_spec_one_worker,
[vllm]                 | ERROR 06-02 05:17:18 [core.py:387]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py", line 699, in get_kv_cache_config
[vllm]                 | ERROR 06-02 05:17:18 [core.py:387]     check_enough_kv_cache_memory(vllm_config, kv_cache_spec, available_memory)
[vllm]                 | ERROR 06-02 05:17:18 [core.py:387]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py", line 545, in check_enough_kv_cache_memory
[vllm]                 | ERROR 06-02 05:17:18 [core.py:387]     raise ValueError(
[vllm]                 | ERROR 06-02 05:17:18 [core.py:387] ValueError: To serve at least one request with the models's max seq len (40960), (10.00 GiB KV cache is needed, which is larger than the available KV cache memory (1.25 GiB). Based on the available memory,  Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine.
[vllm]                 | ERROR 06-02 05:17:18 [core.py:387] 

```

Decreasing the context worked. The bellow is how I managed to add it to the docker-compose.yaml

```bash
cat docker-compose.yaml
...
    command:
      - "--model"
      - "RedHatAI/Qwen3-32B-quantized.w4a16"
      - "--max_model_len"
      - "4096"
...
```


Decrease achieved

```bash
# Checking the size of the models
du -sh ~/.cache/huggingface/hub/*
62G	/home/jubapa/.cache/huggingface/hub/models--Qwen--Qwen3-32B
18G	/home/jubapa/.cache/huggingface/hub/models--RedHatAI--Qwen3-32B-quantized.w4a16
```

