import os
import json

from transformers import AutoTokenizer

batch_size = 16
input_ids_len = int(3.5*1024)

DIR = "dataset/alices-adventures-in-wonderland"

def load_text():
    file_names = os.listdir(DIR)
    data = []
    for name in file_names:
        with open(os.path.join(DIR, name), "r") as f:
            data.append(f.read())
    return data

text_data = load_text()
print(f"[+] unique seqs: {len(text_data)}")
tokenizer = AutoTokenizer.from_pretrained("DeepSeek-V3-Tokenizer")
token_ids_list = [
    tokenizer(
        text_data[i%len(text_data)],
        padding='max_length',
        max_length=input_ids_len,
        truncation=True,
        padding_side='left'
    ).get('input_ids') for i in range(batch_size)
]
print([len(i) for i in token_ids_list])
# print(token_ids_list[0])

prompts = [tokenizer.decode(token_ids, skip_special_tokens=False) for token_ids in token_ids_list]
with open(f"dataset/prompts-bs{batch_size}-seq{input_ids_len}.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(prompts, indent=4))
