# Mutual Information Estimation

This directory contains code for estimating bipartite and two-point mutual information in natural language.

## Contents

- `compute_bmi_direct_with_ngram_correction_from_files.py`: Estimates bipartite mutual information with n-gram correction
- `compute_bmi_vclub_from_files.py`: Implements vCLUB estimator for bipartite mutual information
- `compute_entropy_from_llm.py`: Estimates entropy using llms
- `compute_tmi_from_files_ngram.py`: Estimates two-point mutual information using n-gram statistics
- `count_tokens_marginal_ngram.py`: Counts n-gram statistics for single tokens
- `count_tokens_marginal_ngram_d.py`: Counts token pairs at various distances for two-point mutual information
- `split_dataset_into_chunks.py`: Prepares dataset

## Usage

1. Prepare dataset:
   ```
   python split_dataset_into_chunks.py
2. Calculate the n-gram statistics:
   ```
   # count 1-gram statistics
   python count_token_marginal_ngram.py
   # count 2-gram statistics with tokens separated by a distance d
   python count_token_marginal_ngram_d.py
3. Estimate entropy:
   ```
   python compute entropy_from_llm.py
4. Estimate the bipartite mutual information
   ```
   # using the direct estimator with n-gram correction
   python compute_bmi_direct_with_ngram_correction_from_files.py
   # using the vCLUB estimator
   python compute_bmi_vclub_from_files.py
5. Estimate the two-point mutual information
   ```
   # note the bias-corrected fitting function at the end of the file
   python compute_tmi_from_files_ngram.py
