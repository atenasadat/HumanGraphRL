import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

prompt = "Implement a dynamic programming algorithm in Python to compute the longest common subsequence of two strings. Think step by step inside <think> ... </think> tags before writing code."
messages = [
    {"role": "system", "content": "You are an expert engineer. Always think step by step inside <think> ... </think> tags before providing code."},
    {"role": "user", "content": prompt},
]
inputs = tok(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=250, do_sample=True, temperature=0.7)
res = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("=== SAMPLE GENERATION ===")
print(res[:400])
