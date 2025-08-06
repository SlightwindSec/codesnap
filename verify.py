import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("DeepSeek-V3-Tokenizer")
with open("generated_data_2048_20.json", "r") as f:
    data = json.load(f)

prompts = [tokenizer.encode(i, add_special_tokens=False) for i in data]
print([len(i) for i in prompts])
