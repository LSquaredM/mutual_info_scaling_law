# Mutual info scaling law
# Copyright (C) 2025 Zhuo Chen

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import os
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer
from collections import Counter
from functools import reduce
import numpy as np
import pickle


def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], truncation=False, padding=False, add_special_tokens=False)

def map_count_tokens_at_distance(example, token_key='input_ids', distance=1):
    counter = Counter()
    for seq in example[token_key]:
        seq_len = len(seq)
        # Skip if sequence is shorter than distance
        if distance >= seq_len:
            continue
        # Create pairs of tokens at specified distance
        for i in range(seq_len - distance):
            # Convert pair of tokens to tuple for hashability
            pair = tuple(t for t in (seq[i], seq[i + distance]))
            counter[pair] += 1
    
    token_ids = list(counter.keys())
    counts = list(counter.values())
    return {'token_ids': [token_ids], 'counts': [counts]}

def reduce_batch_counters(examples):
    counter = Counter()
    for ids, counts in zip(examples['token_ids'], examples['counts']):
        counter.update(dict((tuple(id), count) for id, count in zip(ids, counts)))
    token_ids = list(counter.keys())
    counts = list(counter.values())
    return {'token_ids': [token_ids], 'counts': [counts]}

def get_token_marginal_distribution(dataset_dict, tokenizer, distance=1, token_key='input_ids', batch_size=768):
    vocab_size = len(tokenizer)
    
    # Process each split
    final_counter = Counter()
    
    for split_name, dataset in dataset_dict.items():
        tokenized_dataset = dataset.map(
            tokenize_function,
            fn_kwargs={'tokenizer': tokenizer},
            batched=True,
            batch_size=batch_size,
            remove_columns=dataset.column_names,
            num_proc=32,
        )
        
        mapped = tokenized_dataset.map(
            map_count_tokens_at_distance,
            fn_kwargs={'token_key': token_key, 'distance': distance},
            batched=True,
            batch_size=batch_size,
            remove_columns=tokenized_dataset.column_names,
            num_proc=32,
            # keep_in_memory=True,
        )

        reduced = mapped.map(
            reduce_batch_counters,
            batched=True,
            batch_size=batch_size,
            remove_columns=['token_ids', 'counts'],
            num_proc=32,
            keep_in_memory=True,
        )

        reduced = reduced.map(
            reduce_batch_counters,
            batched=True,
            batch_size=batch_size,
            remove_columns=['token_ids', 'counts'],
            num_proc=4,
            keep_in_memory=True,
        )
        
        # Accumulate counts from this split
        for ids, counts in zip(reduced['token_ids'], reduced['counts']):
            final_counter.update(dict((tuple(id), count) for id, count in zip(ids, counts)))

    counts = dict(final_counter)

    return counts



if __name__ == "__main__":

    model_id = "meta-llama/Llama-3.1-405B-FP8"

    dataset_name = "pg19"
    save_dir = f'./long_range_counts/{model_id.replace("/", "_")}/{dataset_name}'
    os.makedirs(save_dir, exist_ok=True)

    if dataset_name == "pg19":
        dataset = load_dataset("deepmind/pg19", trust_remote_code=True)
    elif dataset_name == "wiki":
        dataset = load_dataset("wikimedia/wikipedia", "20231101.en", trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    vocab_size = len(tokenizer)

    distances = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    for d in distances:
        print(f"Processing distance {d}")
        file_name = f"{save_dir}/count_{d}distance.pkl"
        if os.path.exists(file_name):
            print(f"File already exists: {file_name}")
            continue
        counts = get_token_marginal_distribution(dataset, tokenizer, distance=d)
        with open(file_name, "wb") as f:
            pickle.dump(counts, f)
        print(f"Saved counts for distance {d}")

