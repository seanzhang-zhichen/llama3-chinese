import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "zhichen/Llama3-Chinese" # 替换成你自己的模型路径

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)


messages = [
    {"role": "system", "content": "A chat between a curious user and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the user's questions."},
    {"role": "user", "content": "你是谁"},
]


prompt = pipeline.tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True
)

terminators = [
      pipeline.tokenizer.eos_token_id,
      pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
        prompt,
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0.6,
        top_p=1,
        repetition_penalty=1.05
)
print(outputs[0]["generated_text"][len(prompt):])