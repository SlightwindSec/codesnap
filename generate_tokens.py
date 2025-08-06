import argparse
import json
import random
import sys
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from transformers import AutoTokenizer, PreTrainedTokenizer
except ImportError:
    logging.error("Hugging Face's `transformers` library is not installed.")
    logging.error("Please install it using: pip install transformers")
    sys.exit(1)


def generate_single_verified_string(
    tokenizer: PreTrainedTokenizer,
    target_len: int,
    max_retries: int = 50
) -> Optional[str]:
    vocab_size = tokenizer.vocab_size
    initial_len_buffer = target_len + 30

    for _ in range(max_retries):
        random_ids = random.choices(range(vocab_size), k=initial_len_buffer)
        temp_string = tokenizer.decode(
            random_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )

        re_encoded_ids = tokenizer.encode(temp_string, add_special_tokens=False)

        if len(re_encoded_ids) < target_len:
            continue

        truncated_ids = re_encoded_ids[:target_len]

        final_string = tokenizer.decode(
            truncated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )

        final_ids = tokenizer.encode(final_string, add_special_tokens=False)

        if len(final_ids) == target_len:
            return final_string

    logging.warning(
        f"Failed to generate a valid string of length {target_len} after {max_retries} retries. "
        "Consider increasing the buffer or retries if this happens frequently."
    )
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate a batch of random strings with a precise token length.",
        epilog="Example: python generate_exact_length_seq.py meta-llama/Llama-2-7b-hf 128 10"
    )
    parser.add_argument(
        "tokenizer_path",
        type=str,
        help="Path or Hugging Face Hub name for the AutoTokenizer (e.g., 'bert-base-uncased')."
    )
    parser.add_argument(
        "seq_len",
        type=int,
        help="The exact number of tokens the encoded string should have."
    )
    parser.add_argument(
        "batch_size",
        type=int,
        help="The number of random strings to generate."
    )
    
    args = parser.parse_args()

    if args.seq_len <= 0 or args.batch_size <= 0:
        logging.error("Sequence length and batch size must be positive integers.")
        sys.exit(1)

    logging.info(f"Loading tokenizer: {args.tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    except Exception as e:
        logging.error(f"Failed to load tokenizer '{args.tokenizer_path}'. Error: {e}")
        sys.exit(1)

    logging.info(f"Generating {args.batch_size} samples, each with exactly {args.seq_len} tokens...")

    results: List[str] = []
    while len(results) < args.batch_size:
        generated_string = generate_single_verified_string(tokenizer, args.seq_len)
        
        if generated_string is not None:
            results.append(generated_string)
            logging.info(f"Successfully generated sample {len(results)}/{args.batch_size}")

    output_filename = f"generated_data_{args.seq_len}_{args.batch_size}.json"
    logging.info(f"Writing {len(results)} samples to '{output_filename}'...")
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logging.info("Done!")


if __name__ == "__main__":
    main()
