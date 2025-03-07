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
import sys
import warnings
import random
import datetime
import math
import time
from typing import Optional, Union, List, Dict, Any, Tuple
from contextlib import contextmanager
from ruamel.yaml import YAML
import tqdm
import fire
import pickle
import wandb
import numpy as np
import torch
torch.set_float32_matmul_precision('high') # to use tensor cores
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 1000
torch._dynamo.config.accumulated_cache_size_limit = 1000
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import einops

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, PreTrainedModel
from datasets import load_dataset, load_from_disk
from gpt_flex.gpt2_flex_attn import GPT2LMHeadModel

from transformers.trainer_utils import speed_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(model_name, model_type, pretrained=True,
              cache_dir=None, flash_attention=False,
              gpt_dropout=None,
              **kwargs):
    ### gpt2 ###
    if model_type == "gpt2":
        if sys.platform in ["linux", "linux2"] and flash_attention:
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = "eager"
        print(f"Using attention implementation: {attn_implementation}")
        if cache_dir is not None and cache_dir.lower() != 'none':
            config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir, attn_implementation=attn_implementation)
        else:
            config = AutoConfig.from_pretrained(model_name, attn_implementation=attn_implementation)
        if pretrained:
            assert False, "Pretrained models are not tested for our implementation yet"
        else:
            if kwargs != {}:
                print("Modifying the model configuration with the following keyword arguments: " + ", ".join(kwargs.keys()))
                for key, value in kwargs.items():
                    print(f"Setting {key} to {value}")
                    setattr(config, key, value)
            for key in dir(config):
                if 'drop' in key.lower():
                    print(f"Disabling dropout in config: config.{key}")
                    setattr(config, key, 0.0)
            vocab_size = 50432
            config.vocab_size = vocab_size
            model = GPT2LMHeadModel(config)

        if gpt_dropout is not None:
            for module in model.modules():
                for attr_name in dir(module):
                    if "drop" in attr_name.lower():
                        print(attr_name)
                        attr_value = getattr(module, attr_name)
                        if isinstance(attr_value, (float, int)):
                            setattr(module, attr_name, gpt_dropout)
                        if isinstance(attr_value, nn.Dropout):
                            attr_value.p = gpt_dropout

    ### mamba ###
    elif model_type == "mamba":
        if cache_dir is not None and cache_dir.lower() != 'none':
            config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        else:
            config = AutoConfig.from_pretrained(model_name)
        if pretrained:
            if cache_dir is not None and cache_dir.lower() != 'none':
                model = AutoModelForCausalLM.from_pretrained(model_name, config=config, cache_dir=cache_dir)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
        else:
            model = AutoModelForCausalLM.from_config(config)

        # need this to avoid TypeError: 'MambaCache' object is not iterable
        config.use_cache = False
        model.config.use_cache = False
    
    return model, config

def load_model_checkpoint(model, model_checkpoint):
    if model_checkpoint is not None:
        print(f"Loading model checkpoint from {model_checkpoint}")
        model_state_dict = torch.load(model_checkpoint)
        # trim state_dict
        new_model_state_dict = {}
        for k, v in model_state_dict.items():
            new_model_state_dict[k.replace("_orig_mod.", "")] = v
        # rename gpt2 state dict layers (load correct layers and not the custom layers for powerdata training)
        if isinstance(model, GPT2LMHeadModel): #!
            new_model_state_dict["lm_head.weight"] = new_model_state_dict["original_lm_head.weight"] #!
            new_model_state_dict["transformer.wte.weight"] = new_model_state_dict["transformer.original_wte.weight"] #!
        # if positional embedding have different size, truncate or pad with new weights
        try:
            if model.transformer.wpe.weight.size(0) != new_model_state_dict["transformer.wpe.weight"].size(0):
                print(f"Truncating or padding the positional embedding layer from {model.transformer.wpe.weight.size(0)} to {new_model_state_dict['transformer.wpe.weight'].size(0)}")
                new_wpe_weight = model.transformer.wpe.weight.data
                new_wpe_weight[:min(new_wpe_weight.size(0), new_model_state_dict["transformer.wpe.weight"].size(0))] = new_model_state_dict["transformer.wpe.weight"][:min(new_wpe_weight.size(0), new_model_state_dict["transformer.wpe.weight"].size(0))]
                new_model_state_dict["transformer.wpe.weight"] = new_wpe_weight
        except Exception as e:
            print(f"Error in trying to truncate or pad the positional embedding layer: {e}")
        missing_keys, unexpected_keys = model.load_state_dict(new_model_state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")

def get_dataset(dataset_name, dataset_config_name=None, dataset_cache_dir=None, min_seq_len=None, trun_seq_len=None, eval_num_samples=2000, tokenizer=None):
    if dataset_name in ["wiki", "wikipedia"]:
        dataset_name = "../measure_mutual_info/processed_wiki"

    if dataset_name in ["pg19"]:
        dataset_name = "../measure_mutual_info/processed_pg19"

    if dataset_name in ["wiki_5000"]:
        dataset_name = "../measure_mutual_info/processed_wiki_5000"

    if dataset_name in ["pg19_5000"]:
        dataset_name = "../measure_mutual_info/processed_pg19_5000"

    print(f"Loading dataset {dataset_name} with config {dataset_config_name} from cache {dataset_cache_dir}")

    dataset = load_from_disk(dataset_name)


    def filter_by_min_length(example):
        return len(example["text"].split() if "text" in example else example["corpus"].split()) >= min_seq_len
    
    if min_seq_len is not None:
        dataset = dataset.filter(filter_by_min_length, num_proc=48)

    from sympy import nextprime
    prime = nextprime(int(len(dataset) / math.sqrt(eval_num_samples)))
    eval_indices = [(i * prime) % len(dataset) for i in range(eval_num_samples)]
    train_indices = list(set(range(len(dataset))) - set(eval_indices))

    train_dataset = dataset.select(train_indices)
    eval_dataset = dataset.select(eval_indices)

    def tokenize_function(example):
        return tokenizer(
            example["text"] if "text" in example else example["corpus"],
            truncation=True,
            padding="max_length",
            max_length=trun_seq_len,
            add_special_tokens=True, ### somehow gpt (and mamba1) doesn't seem to have special tokens, come back to this later
        )
    if tokenizer is not None:
        train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=48, remove_columns=train_dataset.column_names)
        eval_dataset = eval_dataset.map(tokenize_function, batched=True, num_proc=48, remove_columns=eval_dataset.column_names)
        prime = nextprime(int(len(train_dataset) / math.sqrt(eval_num_samples)))
        train_val_indices = [(i * prime) % len(train_dataset) for i in range(eval_num_samples)]
        train_eval_dataset = train_dataset.select(train_val_indices)

    return train_dataset, eval_dataset, train_eval_dataset

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # Extract additional eval datasets if provided
        self.additional_eval_datasets = kwargs.pop("additional_eval_datasets", {})
        plot_token_level_losses = kwargs.pop("plot_token_level_losses", True)
        super().__init__(*args, **kwargs)
        self.plot_token_level_losses = plot_token_level_losses
        self.reset_token_level_metrics()

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits_shifted = outputs.logits[:, :-1, :].contiguous()
        labels_shifted = inputs["input_ids"][:, 1:].contiguous()
        labels_shifted[inputs["attention_mask"][:, :-1] == 0] = -100
        labels_shifted[inputs["input_ids"][:, :-1] == self.tokenizer.pad_token_id] = -100

        loss = F.cross_entropy(
            logits_shifted.view(-1, logits_shifted.size(-1)),
            labels_shifted.view(-1),
            reduction="mean",
            ignore_index=-100
        )
        
        return (loss, outputs) if return_outputs else loss

    def reset_token_level_metrics(self):
        self.token_loss_sums = None
        self.token_counts = None

    def prediction_step(
        self,
        model,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            outputs = model(**inputs)
            logits_shifted = outputs.logits[:, :-1, :].contiguous()
            labels_shifted = inputs["input_ids"][:, 1:].contiguous()
            attention_mask_shifted = inputs["attention_mask"][:, :-1]
            
            labels_shifted[attention_mask_shifted == 0] = -100
            labels_shifted[inputs["input_ids"][:, :-1] == self.tokenizer.pad_token_id] = -100
            
            token_losses = F.cross_entropy(
                logits_shifted.view(-1, logits_shifted.size(-1)),
                labels_shifted.view(-1),
                reduction="none",
                ignore_index=-100
            ).view(logits_shifted.size(0), -1)
            
            if self.token_loss_sums is None:
                self.token_loss_sums = np.zeros(token_losses.size(1), dtype=np.float64)
                self.token_counts = np.zeros(token_losses.size(1), dtype=np.int64)
            
            valid_tokens = (labels_shifted != -100).cpu().numpy()
            batch_losses = token_losses.cpu().numpy()
            
            self.token_loss_sums += np.sum(batch_losses * valid_tokens, axis=0)
            self.token_counts += np.sum(valid_tokens, axis=0)
            
            valid_token_mask = (labels_shifted != -100).float()
            loss = (token_losses * valid_token_mask).sum() / valid_token_mask.sum()

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, logits_shifted, labels_shifted)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        # Store all metrics
        all_metrics = {}
        all_token_level_metrics = []
        
        # First run evaluation on the primary dataset exactly as in the original code
        self.reset_token_level_metrics()
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )
        all_metrics.update(metrics)
        all_token_level_metrics.append(dict(dataset_name="eval", token_loss_sums=self.token_loss_sums, token_counts=self.token_counts))
        
        # Then evaluate additional datasets if any
        for dataset_name, dataset in self.additional_eval_datasets.items():
            self.reset_token_level_metrics()
            dataset_metrics = super().evaluate(
                eval_dataset=dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=f"{dataset_name}"
            )
            all_metrics.update(dataset_metrics)
            all_token_level_metrics.append(dict(dataset_name=dataset_name, token_loss_sums=self.token_loss_sums, token_counts=self.token_counts))

        self.log_token_level_losses(all_token_level_metrics)
        
        return all_metrics

    def log_token_level_losses(self, all_token_level_metrics):
        """Create and log token-level loss visualization to W&B for a specific dataset"""
        for token_level_metrics in all_token_level_metrics:
            dataset_name = token_level_metrics["dataset_name"]
            token_loss_sums = token_level_metrics["token_loss_sums"]
            token_counts = token_level_metrics["token_counts"]
        
            valid_positions = token_counts > 0
            avg_token_losses = np.zeros_like(token_loss_sums)
            avg_token_losses[valid_positions] = (
                token_loss_sums[valid_positions] / token_counts[valid_positions]
            )
            curr_step = self.state.global_step

            wandb.log({f"{dataset_name}_avg_token_loss": avg_token_losses.mean()}, step=curr_step)

            if self.plot_token_level_losses:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(range(1, len(avg_token_losses)+1), avg_token_losses)
                plt.xlabel("Token Location")
                plt.ylabel("Loss")
                plt.title(f"Token-wise Loss - {dataset_name}")
                plt.xscale("log")
                plt.yscale("log")
                wandb.log({f"{dataset_name}_token_wise_loss_{curr_step}": plt}, step=curr_step)
                plt.close()

def main(
    model_name: str = "mamba-130m-hf",
    tokenizer_name: str = "EleutherAI/gpt-neox-20b", # if None, will default to appropriate known tokenizer
    pretrained: bool = False,
    num_epochs: int = 1,
    batch_size: int = 2,
    grad_accumulation_steps: int = 1,
    dataset_name: str = "pg19_5000",
    dataset_config_name: str = None, # if None, will default to appropriate known dataset
    dataset_cache_dir: str = None, # if None, will default to appropriate known cache dir
    trun_seq_len: int = 256,
    min_seq_len: int = 4096,
    eval_num_samples: int = 10000,
    log_interval: int = 10,
    eval_interval: int = 10,
    save_interval: int = 10000,
    save_dir: str = f"results/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    resume: bool = False,
    model_checkpoint: str = None,
    seed: int = 42,
    grad_norm_clip_value: float = 1.,
    lr: float = 0.0005,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    wd: float = 0.01,
    warmup_steps: int = 200,
    flash_attention: bool = False,
    bf16: bool = False, # must be True to use flash_attention
    plot: bool = True,
    wandb_project: str = "test",
):
    
    all_args = locals()
    set_seed(seed)
    yaml = YAML()
    os.makedirs(save_dir, exist_ok=True)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'args.yaml'), 'w') as f:
            yaml.dump(all_args, f)

    wandb.init(
        project=wandb_project,
        name=save_dir.replace("results/", "").replace("/", "_"),
        config=all_args,
    )

    if model_name.startswith("openai-community/gpt2"):
        model_type = "gpt2"
    elif model_name.startswith("gpt2"):
        model_name = "openai-community/" + model_name
        model_type = "gpt2"
    elif model_name.startswith("state-spaces/mamba"):
        model_type = "mamba"
    elif model_name.startswith("mamba"):
        model_name = "state-spaces/" + model_name
        model_type = "mamba"
    else:
        raise ValueError(f"Model name {model_name} not recognized.")

    model, config = get_model(
        model_name=model_name,
        model_type=model_type,
        pretrained=pretrained,
        cache_dir=None,
        flash_attention=flash_attention,
        max_position_embeddings=trun_seq_len, # this will overwrite the config if not using pretrained
        n_ctx=trun_seq_len, # this will overwrite the config if not using pretrained, althought n_ctx doesn't seem to be used in the model
    )

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_parameters}")
    wandb.log({"num_parameters": num_parameters})

    if model_checkpoint is not None:
        load_model_checkpoint(model, model_checkpoint)

    model = torch.compile(model)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name if tokenizer_name is not None else model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset, eval_dataset, train_eval_dataset = get_dataset(
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        dataset_cache_dir=dataset_cache_dir,
        min_seq_len=min_seq_len,
        trun_seq_len=trun_seq_len,
        eval_num_samples=eval_num_samples,
        tokenizer=tokenizer,
    )

    training_args = TrainingArguments(
        output_dir=save_dir,
        overwrite_output_dir=not resume,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accumulation_steps,
        evaluation_strategy="steps",
        eval_steps=eval_interval,
        save_steps=save_interval,
        logging_dir=f"{save_dir}/logs",
        logging_steps=log_interval,
        bf16=bf16,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        weight_decay=wd,
        warmup_steps=warmup_steps,
        adam_beta1=betas[0],
        adam_beta2=betas[1],
        adam_epsilon=eps,
        max_grad_norm=grad_norm_clip_value,
        load_best_model_at_end=True,
        dataloader_num_workers=48,
        report_to="wandb",
        save_total_limit=3,
        remove_unused_columns=False,
        save_safetensors=False
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        additional_eval_datasets={
            "train_eval": train_eval_dataset
        },
        tokenizer=tokenizer,
        plot_token_level_losses=plot,
    )

    trainer.train(resume_from_checkpoint=resume)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    wandb.finish()

    return

if __name__ == "__main__":
    fire.Fire(main)
