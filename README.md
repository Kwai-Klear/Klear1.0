# Klear1.0

<div align="center">
  <img src="figures/klear-logo-02.png" width="500"/>
  <p>
    🤗 <a href="https://huggingface.co/Kwai-Klear">Hugging Face</a> |  📑 <a href="https://github.com/Kwai-Klear/Klear1.0">Technique Report |  💬 <a href="https://github.com/Kwai-Klear/Klear1.0/issues">Issues & Discussions</a>
  </p>
</div>


## 🔥News

- 2025.09.05: We’ve released the `Klear-46B-A2.5B` series, which currently includes `a base model` and an `instruction-tuned model with DPO`. `A reasoning-enhanced variant is also in training` — stay tuned for upcoming updates!


## 1. Introduction


`Klear-46B-A2.5B` is a sparse Mixture-of-Experts (MoE) large language model developed by **the Kwai-Klear Team at Kuaishou**, designed to deliver both **high performance** and **inference efficiency**. It features **256 experts**, with only **8 experts and 1 shared expert activated** per layer during the forward pass, resulting in **46 billion total parameters** but just **2.5 billion active** — achieving dense-level performance at a fraction of the computational cost.

The model was trained on over **22 trillion tokens** using a **three-stage progressive curriculum**:

**1. Foundational Knowledge Learning (12T tokens):**
General-purpose datasets such as CommonCrawl were processed with stratified quality filters, following a curriculum learning strategy that progresses from lower to higher data quality.

**2. Data Complexity Enhancement (8T tokens):**
The proportion of mathematical, coding, and STEM-related data was gradually increased to strengthen the model's reasoning and problem-solving capabilities.

**3. Reasoning Enhancement and Longcontext Stage (2T tokens):**
Training focused on synthetic and reasoning-intensive data, combined with a fast learning rate annealing strategy to maximize data efficiency and optimize final performance.

As a result, Klear-46B-A2.5B-Base matches or surpasses the performance of dense models with several times more active parameters, while offering significantly better efficiency and cost-effectiveness for real-world deployment.


## Model Summary

The base and instruction tuned + DPO models have the following architecture:

| **propoty**               | **value**                                                              |
|---------------------------|------------------------------------------------------------------------|
| hidden_size       | 2048                                                                     |
| moe_intermediate_size                  | 896                                                                    |
| n_shared_experts       | 1                                                                      |
| num_attention_heads         | 32                                                                    |
| num_experts     | 256                                                                    |
| num_experts_per_tok               | 8                                                                    |
| num_hidden_layers       | 32                                                                      |
| num_key_value_heads               | 4                                                                   |
| vocab_size                | 151936                                                                 |
| tie_word_embeddings       | false                                                                  |
| context length       | 65536                                                                  |


### Model Downloads

<div align="center">

| **Model** | **#Total Params** | **#Activated Params** | **Context Length** | **Download Link** |
| :------------: | :------------: | :------------: | :------------: | :------------: |
| Klear-46B-A2.5B-Base | 46B | 2.5B | 64K   | [🤗 Hugging Face](https://huggingface.co/Kwai-Klear/Klear-46B-A2.5B-Base)   |
| Klear-46B-A2.5B-Instruct  | 46B | 2.5B |  64K   | [🤗 Hugging Face](https://huggingface.co/Kwai-Klear/Klear-46B-A2.5B-Instruct)   |

</div>


## 2. Benchmark Evaluation
### Klear-46B-A2.5B-Base Evaluation Results
| Ability     | Benchmark              | Klear-46B-A2.5B-Base | MiMO-7B-Base | Qwen3-8B-BASE | Qwen3-14B-BASE | Ling-lite-1.5-Base | Qwen3-30B-A3B-BASE |
| ----------- | ---------------------- | -------------------- | ------------ | ------------- | -------------- | ------------------ | ------------------ |
|             | # Total Params         | 46B                  | 7B           | 8B            | 14B            | 16.8B              | 30B                |
|             | # Activated Params     | 2.5B                 | 7B           | 8B            | 14B            | 2.75B              | 3B                 |
| **Code**    | HumanEval (0-shot*)    | 89                   | -            | 84.1          | 87.8           | 83.5               | 90.9               |
|             | MBPP (3-shot)          | 76                   | 69.2*        | 69            | 74             | 66.6               | 75.6               |
| **Math**    | MATH (4-shot, cot)     | 55.7                 | 38.8         | 60.8*         | 62.02*         | 59.9               | 59.04*             |
|             | CMATH (3-shot)         | 87.83                | 78.5         | 88.3          | 90.7           | 85.7               | 89.7               |
|             | GSM8K (4-shot, cot)    | 87.3                 | 78.47        | 89.4          | 90.3           | 87.6               | 91.1               |
| **General** | MMLU-Pro (5-shot, cot) | 57.6                 | 43.1         | 55.2          | 58.1           | 49.9               | 58.8               |
|             | MMLU (5-shot)          | 80.5                 | 69.24        | 77.1          | 80.6           | 73.7               | 80.4               |
|             | CEval (5-shot)         | 89.8                 | 67.98        | 81.9          | 84.8           | 78.2               | 87.4               |
|             | CMMLU (5-shot)         | 88                   | 70.79        | 82            | 85.6           | 81.2               | 87.1               |
|             | GPQA (0-shot)          | 35.3                 | 31.03        | 33.9          | 35.7           | 30.1               | 35.5               |
|             | AGIEval (0-shot)       | 52.3                 | 48.3*        | 51.7          | 55.7           | 54.3               | 56                 |
|             | BBH (3-shot, cot)      | 77.9                 | 75.6         | 78.1          | 80.1           | 75.4               | 81.2               |
|             | HellaSwag (0-shot)     | 80.5                 | 80*          | 78.7          | 81.5           | 80                 | 81.2               |
|             | Triviaqa (5-shot)      | 69.6                 | 60.8*        | 56.3          | 62.1           | 60.9               | 65.6               |
|             | Naturalqs (5-shot)     | 37.5                 | 23.46        | 25.7          | 29.1           | 28                 | 30.7               |
|             | PIQA (0-shot)          | 81.6                 | 80.14        | 79.5          | 81.9           | 82                 | 80.7               |
|             | OpenBookQA (0-shot)    | 37.8                 | 34.2         | 35            | 35.6           | 38.2               | 34.6               |
|             | Average                | 69.66                | -            | 66.62         | 69.60          | 65.60              | 70.41              |

Note:
1. `*`During pretraining, we found that the HumanEval metric fluctuated significantly and was extremely sensitive to formatting. Therefore, we referred to the prompt from Ling-series paper to modify the original HumanEval. The results in the table are the evaluation metrics after this modification. 
2. For Mimo-base-7B, the results marked with `*` are sourced from their public report, other evaluations are conducted based on internal evaluation frameworks.

### Klear-46B-A2.5B-Instruct Evaluation Results
| Ability       | Benchmark                   | Klear-46B-A2.5B-Instruct | InternLM3-8B-Instruct | MiniCPM4-8B | Qwen3-8B (NoThink) | gemma3-12b-it | Phi4-14B | Qwen3-30B-A3B-2507 |
| ------------- | --------------------------- | --------------- | --------------------- | ----------- | ------------------ | ------------- | -------- | ------------------ |
|               | # Total Params              | 46B             | 8B                    | 8B          | 8B                 | 12B           | 14B      | 30B                |
|               | # Activated Params          | 2.5B            | 8B                    | 8B          | 8B                 | 12B           | 14B      | 3B                 |
| **General**   | MMLU-Redux                  | 81.61           | 74.65                 | 77.63       | 79.32              | 78.39         | 83.09    | 88.11              |
|               | MMLU-Pro                    | 63.47           | 50.87                 | 54.69       | 63.8               | 60.69         | 67.25    | 78.22              |
|               | GPQA-Diamoind               | 47.85           | 38.76                 | 38.51       | 51.77              | 39.02         | 59.47    | 71.21              |
|               | SimpleQA                    | 6.52            | 4.44                  | 3.51        | 5.5                | 6.22          | 3.28     | 23.39              |
|               | CLUEWSC                     | 88.16           | 77.63                 | 81.91       | 82.89              | 91.12         | 88.16    | 92.11              |
|               | CEval                       | 83.99           | 84.26                 | 81.78       | 81.66              | 60.81         | 64.79    | 88.57              |
|               | C-SimpleQA                  | 42.3            | 25.87                 | 23.13       | 37.07              | 28.97         | 24.77    | 75.37              |
|               | LiveBench 1125              | 50.1            | 26.3                  | 25.5        | 52.1               | 43.1          | 40       | 68.4               |
| **Math**      | MATH500                     | 82.8            | 68.4                  | 79.8        | 85                 | 86.8          | 80.6     | 97.2               |
|               | AIME24                      | 25.62           | 11.25                 | 22.92       | 28.33              | 23.96         | 15.83    | 75                 |
|               | AIME25                      | 18.12           | 8.12                  | 15.21       | 20.62              | 18.33         | 18.75    | 61.88              |
| **Code**      | HumanEval                   | 87.8            | 82.3*                 | 74.39       | 83.54              | 82.32         | 85.37    | 81.71              |
|               | HumanEval+                  | 81.1            | -                     | 70.12       | 76.83              | 75.61         | 83.54    | 76.83              |
|               | MBPPEvalplus                | 83.1            | 62.4                  | 82          | 76.2               | 85.7          | 77.5     | 89.4               |
|               | MBPPEvalplus++              | 70.4            | 50.4                  | 69.3        | 66.1               | 74.1          | 66.7     | 75.1               |
|               | LiveCodeBench v5(2408-2501) | 28.67           | 14.7                  | 12.19       | 27.24              | 24.73         | 23.66    | 41.22              |
| **Alignment** | IF-Eval                     | 80.04           | 79.3                  | 73.01       | 84.47              | 81.52         | 59.33    | 83.92              |
|               | Multi-IF(en+zh)             | 78.73           | 61.83                 | 61.79       | 78.95              | 76.56         | 62.7     | 77.75              |
|               | MTBench                     | 8.23            | 7.86                  | 6.875       | 8.21               | 8.675         | 8.625    | 9.33               |
|               | MT-Eval                     | 8.11            | 7.36                  | 6.7         | 8.18               | 8.45          | 8.12     | -                  |
|               | AlignBench v1.1             | 6.85            | 6.13                  | 5.99        | 6.95               | 6.3           | 6.33     | 7.06               |
|               | Average                     | 53.50           | -                     | 46.05       | 52.61              | 50.54         | 48.95    | -                  |

Note:
1. For InternLM3-8B-Instruct, the results marked with `*` are sourced from their official website, other evaluations are conducted based on internal evaluation frameworks.
2. For Multi-IF, we report the overall average computed across all three rounds, pooling the Chinese and English metrics.

## 3. Quick start

### Inference with huggingface

You can now inference in Transformers starting from version `4.56.0`.

#### Klear-46B-A2.5B-Base

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/path/to/Klear-Base"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", dtype=torch.bfloat16, trust_remote_code=True)

text = "世界上最大的湖是"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=256)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

#### Klear-46B-A2.5B-Instruct

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_path = "/path/to/Klear-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", dtype=torch.bfloat16, trust_remote_code=True)

messages = [
    {"role": "user", "content": "帮我用 python 写一个计算器的代码吧。"}
]
input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
outputs = model.generate(input_tensor.to(model.device), max_new_tokens=1024)

result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
print(result)
```

### Inference with vllm

[vLLM](https://github.com/vllm-project/vllm) is a high-speed and memery-efficicent inference framework. We provide **our own forked version of [vLLM](https://github.com/Kwai-Klear/vllm) here.**

```shell
git clone https://github.com/Kwai-Klear/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip install --editable .
vllm serve /path/to/Klear-Instruct --port 8000 --tensor-parallel-size 8 --trust-remote-code
```

An OpenAI-compatible API will be available at `http://localhost:8000/v1`.

Or you can refer to the following Python script for offline inference
```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_path = "/path/to/Klear-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

llm = LLM(
    model=model_path,
    trust_remote_code=True,
    tensor_parallel_size=torch.cuda.device_count(),
    gpu_memory_utilization=0.7
)
messages = [
    {"role": "user", "content": "帮我用 python 写一个计算器的代码吧。"}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

sampling_params = SamplingParams(
    temperature=0.6, top_p=0.8, max_tokens=512
)

outputs = llm.generate([prompt], sampling_params)

print(outputs[0].outputs[0].text)

```
