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


import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
import pickle
import os
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataset(dataset_name, dataset_config_name=None, dataset_cache_dir=None, min_seq_len=None, trun_seq_len=None, eval_num_samples=2000, tokenizer=None):
    
    dataset = load_from_disk(dataset_name)

    def tokenize_dataset(dataset):
        tokenized = tokenizer(
            dataset["text"] if "text" in dataset else dataset["corpus"],
            truncation=True,
            padding="max_length",
            max_length=trun_seq_len,
            add_special_tokens=True,
        )
        return {'input_ids': tokenized['input_ids'], 'attention_mask': tokenized['attention_mask']}
    
    dataset = dataset.map(tokenize_dataset, num_proc=48, remove_columns=dataset.column_names)

    print(len(dataset))

    dataset = dataset.filter(lambda x: all(x['attention_mask']), num_proc=48)
    print(len(dataset))

    def create_random_pairs(dataset):
        # Get number of pairs we can make
        from sympy import nextprime
        prime = nextprime(int(len(dataset) ** 0.5))
        print(prime)
        print(len(dataset))
        n_pairs = len(dataset) // 2
        indices = [(i * prime + 1234) % len(dataset) for i in range(len(dataset))]
        first_half = dataset.select(indices[:n_pairs])
        second_half = dataset.select(indices[n_pairs:2*n_pairs])
        
        def combine_samples(sample1, sample2):
            return {
                **{f"{k}_1": v for k, v in sample1.items()},
                **{f"{k}_2": v for k, v in sample2.items()}
            }
        
        paired_dataset = first_half.map(
            lambda batch, idx: combine_samples(batch, second_half[idx]),
            with_indices=True,
            num_proc=48,
            remove_columns=first_half.column_names
        )
        
        return paired_dataset

    dataset = create_random_pairs(dataset)
    return dataset

SPLIT_AT_INDICES = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

SAMPLING_PARAMS = SamplingParams(
    temperature=1, 
    top_p=1,
    top_k=-1,
    min_p=0,
    seed=42,
    max_tokens=1,
    min_tokens=1,
    logprobs=0,
    prompt_logprobs=0,
    detokenize=True,
    )

def eval_batch(llm, batch_token_ids):
    outputs = llm.generate(prompt_token_ids=batch_token_ids.tolist(), sampling_params=SAMPLING_PARAMS)
    batch_prompt_log_probs = []
    for output in outputs:
        prompt_log_probs = []
        for item in output.prompt_logprobs[1:]:
            assert len(item) == 1
            for k, v in item.items():
                prompt_log_probs.append(v.logprob)
        batch_prompt_log_probs.append(prompt_log_probs)
    return batch_prompt_log_probs
    
def eval_batch_splits(llm, batch):
    results = {}
    results['bos_token'] = batch['input_ids_1'][0, 0].item()
    results['input_ids_1'] = batch['input_ids_1'][:, 1:].tolist() # Exclude the bos token
    results['input_ids_2'] = batch['input_ids_2'][:, 1:].tolist() # Exclude the bos token
    results['no_split'] = eval_batch(llm, batch['input_ids_1'])
    
    for split in SPLIT_AT_INDICES:
        input_x1 = batch['input_ids_1'][:, :split + 1] # +1 to include the bos token
        input_y2 = batch['input_ids_2'][:, split + 1:] # +1 to exclude the bos token
        results[f'split_{split}'] = eval_batch(llm, torch.cat([input_x1, input_y2], dim=1))
        results[f'marginal_{split}'] = eval_batch(llm, torch.cat([input_x1[:, :1], batch['input_ids_2'][:, split + 1:]], dim=1))

    return results

if __name__ == "__main__":

    set_seed(42)

    model_id = "meta-llama/Llama-3.1-405B-FP8"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    dataset_name = "pg19_5000"
    dataset = get_dataset(dataset_name, min_seq_len=2049, trun_seq_len=2049, tokenizer=tokenizer)

    def custom_collate_fn(batch):
            collated_dict = {}
            
            for key in ['input_ids_1', 'input_ids_2', 'attention_mask_1', 'attention_mask_2',]:
                collated_dict[key] = torch.tensor([item[key] for item in batch])
                    
            return collated_dict
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=32, collate_fn=custom_collate_fn)

    llm = LLM(model=model_id, tensor_parallel_size=8)

    save_dir = f'./{model_id.replace("/", "_")}/{dataset_name}'
    os.makedirs(save_dir, exist_ok=True)
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}")
        if os.path.exists(f'{save_dir}/batch_{i}.pkl'):
            continue
        results = eval_batch_splits(llm, batch)
        with open(f'{save_dir}/batch_{i}.pkl', 'wb') as f:
            pickle.dump(results, f)
        if i > 30:
            break
