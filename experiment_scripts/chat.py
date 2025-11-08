#!/usr/bin/env python3
"""Simple chat script for inference with the trained model."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
MODEL_PATH = "/n/netscratch/dam_lab/Lab/sqin/olmo/checkpoints/OLMo2-1B-stage1/step7000-hf"
print(f"Loading model from {MODEL_PATH}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
# Use the official OLMo-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B")
print("Model loaded!\n")

# Chat loop
while True:
    prompt = input("You: ")
    if prompt.lower() in ["exit", "quit"]:
        break

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Model: {response}\n")
