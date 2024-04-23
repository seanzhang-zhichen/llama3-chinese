<p align="left">
    <a href="README_CN.md">中文</a>&nbsp ｜ &nbspEnglish
</p>
<br><br>

<p align="center">
<a href='https://huggingface.co/spaces/zhichen'>
<img src='./images/logo.png'>
</a>
</p>

<div align="center">
  <p align="center">
    <h3> Llama3-Chinese </h3>

<p align="center">
      <a href='https://huggingface.co/zhichen'>
        <img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Llama3%20Chinese-yellow'>
      </a>
      <a href='https://modelscope.cn/profile/seanzhang'>
        <img src='https://img.shields.io/badge/🤖 ModelScope-Llama3%20Chinese-blue'>
      </a>
      <br>
      <a href=href="https://github.com/seanzhang-zhichen/llama3-chinese/stargazers">
        <img src="https://img.shields.io/github/stars/seanzhang-zhichen/llama3-chinese?color=ccf">
      </a>
      <a href="https://github.com/seanzhang-zhichen/llama3-chinese/blob/main/LICENSE">
        <img alt="GitHub Contributors" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" />
      </a>
</p>
</div>

## Introduce

**Llama3-Chinese** is a large model trained on 500k high-quality Chinese multi-turn SFT data, 100k English multi-turn SFT data, and 2k single-turn self-cognition data, using the training methods of [DORA](https://arxiv.org/pdf/2402.09353.pdf) and [LORA+](https://arxiv.org/pdf/2402.12354.pdf) based on **Meta-Llama-3-8B** as the base.

**Github:** [https://github.com/seanzhang-zhichen/llama3-chinese](https://github.com/seanzhang-zhichen/llama3-chinese)

![DEMO](./images/web_demo.png)


## Download Model


| Model             | Download  |
|:-------------------:|:-----------:|
| Meta-Llama-3-8B        |[ 🤗 HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3-8B) [  🤖 ModelScope](https://modelscope.cn/models/LLM-Research/Meta-Llama-3-8B)|
| Llama3-Chinese-Lora           |[ 🤗 HuggingFace](https://huggingface.co/zhichen/Llama3-Chinese-Lora) [  🤖 ModelScope](https://modelscope.cn/models/seanzhang/Llama3-Chinese-Lora)|
| Llama3-Chinese (merged model)           |[ 🤗 HuggingFace](https://huggingface.co/zhichen/Llama3-Chinese) [  🤖 ModelScope](https://modelscope.cn/models/seanzhang/Llama3-Chinese)|


## Merge LORA Model (Skippable)

1、Download [Meta-Llama-3-8B](https://modelscope.cn/models/LLM-Research/Meta-Llama-3-8B)

```bash
git clone https://www.modelscope.cn/LLM-Research/Meta-Llama-3-8B.git
```

2、Download [Llama3-Chinese-Lora](https://www.modelscope.cn/models/seanzhang/Llama3-Chinese-Lora)

**From ModelScope**
```bash
git lfs install
git clone https://www.modelscope.cn/seanzhang/Llama3-Chinese-Lora.git
```

**From HuggingFace**
```bash
git lfs install
git clone https://huggingface.co/zhichen/Llama3-Chinese-Lora
```

3、Merge Model

```bash
python merge_lora.py \
    --base_model path/to/Meta-Llama-3-8B \
    --lora_model path/to/lora/Llama3-Chinese-Lora  \
    --output_dir ./Llama3-Chinese
```


## Download Llama3-Chinese (Merged Model)

**From ModelScope**
```bash
git lfs install
git clone https://www.modelscope.cn/seanzhang/Llama3-Chinese.git
```

**From HuggingFace**
```bash
git lfs install
git clone https://huggingface.co/zhichen/Llama3-Chinese
```

## Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "zhichen/Llama3-Chinese"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你好"},
]

input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

outputs = model.generate(
    input_ids,
    max_new_tokens=2048,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
```

## CLI DEMO

```bash
python cli_demo.py --model_path zhichen/Llama3-Chinese
```

## WEB DEMO

```bash
python web_demo.py --model_path zhichen/Llama3-Chinese
```


## VLLM WEB DEMO

1、Use [vllm](https://github.com/vllm-project/vllm) deploy model

```bash
python -m vllm.entrypoints.openai.api_server --served-model-name Llama3-Chinese --model ./Llama3-Chinese(Replace it with your own merged model path)
```

2、This command is executed on the CLI

```bash
python vllm_web_demo.py --model Llama3-Chinese
```

## Train Dataset

[deepctrl-sft-data](https://modelscope.cn/datasets/deepctrl/deepctrl-sft-data)


## LICENSE

This project can only be used for research purposes, and the project developer shall not bear any harm or loss caused by the use of this project (including but not limited to data, models, codes, etc.). For details, please refer to [DISCLAIMER](https://github.com/seanzhang-zhichen/Llama3-Chinese/blob/main/DISCLAIMER)。

The License agreement of the Llama3-Chinese project code is the [Apache License 2.0](./LICENSE). The code is free for commercial use, and the model weights and data can only be used for research purposes. Please attach a link to Llama3-Chinese and the licensing agreement in the product description.


## Citation

If you used Llama3-Chinese in your research, cite it in the following format:


```latex
@misc{Llama3-Chinese,
  title={Llama3-Chinese},
  author={Zhichen Zhang, Xin LU, Long Chen},
  year={2024},
  howpublished={\url{https://github.com/seanzhang-zhichen/llama3-chinese}},
}
```

## Acknowledgement

[meta-llama/llama3](https://github.com/meta-llama/llama3)
<br>
[hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=seanzhang-zhichen/Llama3-Chinese&type=Date)](https://star-history.com/#seanzhang-zhichen/Llama3-Chinese&Date)
