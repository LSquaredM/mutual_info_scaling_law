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


from datasets import Dataset, load_dataset, concatenate_datasets
from typing import List, Optional, Dict
from multiprocessing import cpu_count
import nltk
from nltk.tokenize import sent_tokenize

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK's Punkt tokenizer."""
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        raise ImportError("Please install nltk: pip install nltk")
        
    if not text or not text.strip():
        return []
        
    return sent_tokenize(text.strip())

def split_into_corpuses(text: str, min_words: int = 1000) -> List[str]:
    """Split text into corpuses of minimum word length."""
    sentences = split_into_sentences(text)
    
    corpuses = []
    current = []
    word_count = 0
    
    for sentence in sentences:
        words = len(sentence.split())
        
        if word_count >= min_words and current:
            corpuses.append(' '.join(current))
            current = []
            word_count = 0
            
        current.append(sentence)
        word_count += words
    
    return corpuses


def process_text_with_indices(example: Dict, idx: List[int], text_column: str, min_words: int) -> Dict:
    """Process a single text and return individual samples for each corpus."""
    assert len(idx) == 1, "Expected a single index"
    corpuses = split_into_corpuses(example[text_column][0], min_words)
    
    # Return a dictionary where each field is a list of the same length as corpuses
    return {
        "corpus": [c for c in corpuses],  # Individual corpus
        "text_idx": [idx[0]] * len(corpuses),  # Index of original text
        "corpus_idx": list(range(len(corpuses)))  # Index within the original text
    }

def process_and_save_dataset(
    dataset_name: str,
    text_column: str = "text",
    min_words: int = 1000,
    splits: Optional[List[str]] = None,
    num_proc: Optional[int] = None,
    cache_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    dataset_config: Optional[str] = None
) -> Dataset:
    """
    Load and process dataset with parallel processing.
    """
    if num_proc is None:
        num_proc = max(1, cpu_count() - 1)
    
    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config, cache_dir=cache_dir, trust_remote_code=True)
    
    # If no splits specified, use all available splits
    if splits is None:
        splits = list(dataset.keys())
    
    # Process each split
    processed_splits = []
    for split_name in splits:
        if split_name not in dataset:
            print(f"Warning: split '{split_name}' not found in dataset. Available splits: {list(dataset.keys())}")
            continue
        
        # Process with indices and get individual samples
        processed = dataset[split_name].map(
            function=lambda example, idx: process_text_with_indices(example, idx, text_column, min_words),
            with_indices=True,
            batched=True,
            batch_size=1,
            num_proc=num_proc,
            remove_columns=dataset[split_name].column_names,
            desc=f"Processing {split_name} split"
        )
        
        # Remove the extra list nesting from the map operation
        processed = processed.flatten()
        
        # Add split information
        processed = processed.add_column("split", [split_name] * len(processed))
        processed_splits.append(processed)
    
    # Combine all processed splits
    if not processed_splits:
        raise ValueError(f"No valid splits found in dataset. Available splits: {list(dataset.keys())}")
        
    combined_dataset = concatenate_datasets(processed_splits)
    
    if output_path:
        combined_dataset.save_to_disk(output_path)
    
    return combined_dataset


if __name__ == "__main__":
    compbined_pg19 = process_and_save_dataset(
        "deepmind/pg19",
        splits=["train", "validation", "test"],
        text_column="text",
        min_words=1000,
        output_path="./processed_pg19",
        num_proc=48
    )

    compbined_wiki = process_and_save_dataset(
        "wikimedia/wikipedia",
        dataset_config="20231101.en",
        splits=["train"],
        text_column="text",
        min_words=1000,
        output_path="./processed_wiki",
        num_proc=48
    )

    compbined_pg19 = process_and_save_dataset(
        "deepmind/pg19",
        splits=["train", "validation", "test"],
        text_column="text",
        min_words=5000,
        output_path="./processed_pg19_5000",
        num_proc=48
    )

    compbined_wiki = process_and_save_dataset(
        "wikimedia/wikipedia",
        dataset_config="20231101.en",
        splits=["train"],
        text_column="text",
        min_words=5000,
        output_path="./processed_wiki_5000",
        num_proc=48
    )



