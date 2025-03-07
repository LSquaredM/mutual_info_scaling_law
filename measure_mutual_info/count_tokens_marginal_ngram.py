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


from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer
from collections import Counter
from functools import reduce
import numpy as np
import pickle


def tokenize_function(examples, tokenizer):
    return tokenizer(examples['corpus'], truncation=False, padding=False, add_special_tokens=False)

def map_count_tokens(example, token_key='input_ids', n_gram=1):
    counter = Counter()
    for seq in example[token_key]:
        # Create overlapping n-grams
        for i in range(len(seq) - n_gram + 1):
            # Convert sequence of n tokens to tuple for hashability
            ngram = tuple(str(t) for t in seq[i:i + n_gram])
            counter[ngram] += 1
    
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

def get_token_marginal_distribution(dataset, tokenizer, n_gram=1, token_key='input_ids', batch_size=1000):
    vocab_size = len(tokenizer)
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        fn_kwargs={'tokenizer': tokenizer},
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names,
        num_proc=64,
    )
    
    mapped = tokenized_dataset.map(
        map_count_tokens,
        fn_kwargs={'token_key': token_key, 'n_gram': n_gram},
        batched=True,
        batch_size=batch_size,
        remove_columns=tokenized_dataset.column_names,
        num_proc=64,
    )

    reduced = mapped.map(
        reduce_batch_counters,
        batched=True,
        batch_size=batch_size,
        remove_columns=['token_ids', 'counts'],
        num_proc=64,
    )
    
    final_counter = Counter()
    for ids, counts in zip(reduced['token_ids'], reduced['counts']):
        final_counter.update(dict((tuple(id), count) for id, count in zip(ids, counts)))

    # Convert string tuples back to int tuples when creating probabilities
    counts = {
        tuple(int(t) for t in token_id): count
        for token_id, count in sorted(final_counter.items(), key=lambda x: tuple(int(t) for t in x[0]))
    }

    return counts




if __name__ == "__main__":

    model_id = "meta-llama/Llama-3.1-405B-FP8"
    dataset_name = "pg19_5000"
    save_dir = f'./{model_id.replace("/", "_")}/{dataset_name}'

    if dataset_name == "pg19":
        dataset = load_from_disk("./processed_pg19")
    elif dataset_name == "wiki":
        dataset = load_from_disk("./processed_wiki")
    elif dataset_name == "pg19_5000":
        dataset = load_from_disk("./processed_pg19_5000")
    elif dataset_name == "wiki_5000":
        dataset = load_from_disk("./processed_wiki_5000")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    vocab_size = len(tokenizer)

    # Compute for different n-grams
    for n in range(1, 3):  # This will compute for 1-gram and 2-gram
        counts = get_token_marginal_distribution(dataset, tokenizer, n_gram=n)
        with open(f"{save_dir}/count_{n}gram.pkl", "wb") as f:
            pickle.dump(counts, f)
