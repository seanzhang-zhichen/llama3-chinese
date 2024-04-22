# -*- coding: utf-8 -*-

import torch
import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default=None, required=True, type=str,
                        help="Base model name or path")
    parser.add_argument('--lora_model', default=None, required=True, type=str,
                        help="Please specify LoRA model to be merged.")
    parser.add_argument('--output_dir', default='./merged', type=str)
    args = parser.parse_args()

    base_model_path = args.base_model
    lora_model_path = args.lora_model
    output_dir = args.output_dir
    print(f"Base model: {base_model_path}")
    print(f"LoRA model: {lora_model_path}")

    print("Loading LoRA for causal language model")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)


    new_model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        safe_serialization=False
    )
    
    print(f"Merging with merge_and_unload...")
    base_model = new_model.merge_and_unload()

    print("Saving to Hugging Face format...")
    tokenizer.save_pretrained(output_dir)
    base_model.save_pretrained(output_dir, safe_serialization=False)  # max_shard_size='10GB'
    print(f"Done! model saved to {output_dir}")


if __name__ == '__main__':
    main()
