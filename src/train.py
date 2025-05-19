# Adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py

#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import wandb
import logging
import math
import json
import os
import sys
from dataclasses import dataclass, field
from utils.data_analysis import calculate_diversity_for_sample
from typing import Optional
import datasets
import evaluate
import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from utils.data_selection_strategy import *
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_xla_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from utils.utils import preprocess_and_tokenize_data

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.48.0.dev0")

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    iteration: Optional[int] = field(
        default=None, metadata={"help": "The iteration number)."}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    experiment_path: Optional[str] = field(
        default=None,
        metadata={"help": "Experiment file path"},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=True, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    loss_on_last_n_tokens: int = field(
        default=32,
        metadata={
            "help": "Define number of tokens at the end of the sequence to compute the loss on."
        },
    )
    data_selection_strategy: str = field(
        default=None,
        metadata={"help": "Data selection strategy"},
    )
    accumulate_ai_data: bool = field(
        default=False,
        metadata={"help": "Whether to accumulate all previous generations data"}
    )
    human_data_alpha: float = field(
        default=0.0,
        metadata={"help": "Portion of human data to add for training"}
    )
    ai_beta: float = field(
        default=1.0,
        metadata={"help": "Portion of previous generations ai data to use for training"}
    )

    human_data_filepath: Optional[str] = field(default=None, metadata={"help": "Path to the human data file."})

    max_repeats: Optional[int] = field(
        default=10,
        metadata={"help": "Maximum number of times a sample can be repeated in data selection"}
    )

    bias_factor: Optional[float] = field(
        default=10,
        metadata={"help": "Factor to control bias in data selection"}
    )

    upsample_factor: Optional[float] = field(
        default=1.5,
        metadata={"help": "Factor to control upsampling in data selection"}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.test_file is None:
            raise ValueError("Need either a dataset name or a training/test file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`test_file` should be a csv, a json or a txt file."


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            raise ValueError(
                f"Checkpoint detected, set resume_from_checkpoint"
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=None,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
        assert "test" in raw_datasets.keys(), "test dataset needs to be defined"

    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.test_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks

        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            token=model_args.token,
            **dataset_args,
        )            

        assert "test" in raw_datasets.keys(), "test dataset needs to be defined"

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # get column names
    train_column_names = list(raw_datasets["train"].features)
    test_column_names = list(raw_datasets["test"].features)

    if "cls_score" in train_column_names:
        train_column_names.remove("cls_score")
    if "cls_confidence" in train_column_names:
        train_column_names.remove("cls_confidence")

    lm_datasets_train = preprocess_and_tokenize_data(raw_datasets=raw_datasets["train"],
                                                tokenizer=tokenizer,
                                                column_names=train_column_names,
                                                max_length=data_args.block_size,
                                                preprocessing_num_workers=data_args.preprocessing_num_workers,
                                                overwrite_cache=data_args.overwrite_cache)
    print(lm_datasets_train[1]['input_ids'])
    
    if data_args.ai_beta < 1.0:
        lm_datasets_train = lm_datasets_train.select(range(round(len(lm_datasets_train)*data_args.ai_beta)))

    compute_budget = len(lm_datasets_train)
    print(compute_budget)

    lm_datasets_test = preprocess_and_tokenize_data(raw_datasets=raw_datasets["test"],
                                                tokenizer=tokenizer,
                                                column_names=test_column_names,
                                                max_length=data_args.block_size,
                                                preprocessing_num_workers=data_args.preprocessing_num_workers,
                                                overwrite_cache=data_args.overwrite_cache)

    lm_datasets = datasets.DatasetDict({
        'train': lm_datasets_train,
        'test': lm_datasets_test
    })

    if data_args.accumulate_ai_data and data_args.iteration > 1:

        historical_datasets = []

        for iteration in range(1, data_args.iteration):

            print("Iteration", iteration)

            iteration_filepath = f"{data_args.experiment_path}/{iteration}/data.json"

            with open(iteration_filepath, "r", encoding="utf-8") as f:
                iteration_data = json.load(f)

            print("Iteration data", len(iteration_data))

            iteration_raw_dataset = Dataset.from_list(iteration_data)

            print("Iteration dataset", iteration_raw_dataset)

            iteration_dataset = preprocess_and_tokenize_data(raw_datasets=iteration_raw_dataset,
                                                                tokenizer=tokenizer,
                                                                column_names=train_column_names,
                                                                max_length=data_args.block_size,
                                                                preprocessing_num_workers=data_args.preprocessing_num_workers,
                                                                overwrite_cache=data_args.overwrite_cache)

            historical_datasets.append(iteration_dataset)

        total_size = len(lm_datasets["train"])

        lm_datasets_subset = lm_datasets["train"].shuffle(seed=training_args.seed).select(range(int(total_size*0.5)))
        historical_data = concatenate_datasets(historical_datasets)
        historical_data_subset = historical_data.shuffle(seed=training_args.seed).select(range(int(total_size*0.5)))
        lm_datasets["train"] = concatenate_datasets([lm_datasets_subset, historical_data_subset])

    if data_args.human_data_alpha > 0.0:
        assert os.path.exists(data_args.human_data_filepath), "Human data not found:"

        data_files = {}
        dataset_args = {}
        data_files["train"] = data_args.human_data_filepath
        extension = (data_args.human_data_filepath.split(".")[-1])
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        
        human_dataset = load_dataset(
            extension,
            data_files=data_files,
            token=model_args.token,
            **dataset_args,
        )

        human_column_names = list(human_dataset["train"].features)

        if "cls_score" in human_column_names:
            human_column_names.remove("cls_score")
        if "cls_confidence" in human_column_names:
            human_column_names.remove("cls_confidence")


        human_lm_dataset_train = preprocess_and_tokenize_data(raw_datasets=human_dataset["train"],
                                                    tokenizer=tokenizer,
                                                    column_names=human_column_names,
                                                    max_length=data_args.block_size,
                                                    preprocessing_num_workers=data_args.preprocessing_num_workers,
                                                    overwrite_cache=data_args.overwrite_cache)
        
        human_lm_dataset = datasets.DatasetDict({
            'train': human_lm_dataset_train,
        })
        
        human_data_size = len(human_lm_dataset["train"])

        print("Human data size before selection: ", human_data_size)

        # get subset of human data
        final_human_dataset = human_lm_dataset["train"].select(range(round(human_data_size*data_args.human_data_alpha)))

        print("Final Human data size: ", len(final_human_dataset))

        final_human_dataset = final_human_dataset.map(lambda x: {**x, "gt_cls_score": 0})
        lm_datasets = lm_datasets.map(lambda x: {**x, "gt_cls_score": 1})

        lm_datasets["train"] = concatenate_datasets([lm_datasets["train"], final_human_dataset])
        print("After adding human samples", len(lm_datasets["train"]))      
        
    if str(data_args.data_selection_strategy) != 'None':

        data_selection_args = {
            "dataset": lm_datasets["train"], 
            "seed": training_args.seed,
            "max_repeats": data_args.max_repeats,
            "bias_factor": data_args.bias_factor,
            "upsample_factor": data_args.upsample_factor
            }
        
        logger.info(f"Selecting data using {data_args.data_selection_strategy}")
        lm_datasets["train"] = eval(data_args.data_selection_strategy)(**data_selection_args)

    if training_args.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "test" not in lm_datasets:
            raise ValueError("--do_eval requires a test dataset")
        eval_dataset = lm_datasets["test"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    def custom_loss_fn(outputs, labels, num_items_in_batch=None):
        """
        Custom loss function that only computes the loss for the last n tokens.
        """
        # Get the logits (model outputs)
        logits = outputs.logits

        labels = labels.to(logits.device)
        
        # Shift logits and labels to align for causal language modeling
        shift_logits = logits[..., :-1, :].contiguous()  # Shift logits to the left (predict next token)
        shift_labels = labels[..., 1:].contiguous()     # Shift labels to the left (exclude the first token)

        shift_logits = shift_logits[..., -data_args.loss_on_last_n_tokens:, :].contiguous()  # Take the last n tokens
        shift_labels = shift_labels[..., -data_args.loss_on_last_n_tokens:].contiguous()    # Take the last n labels

        # Compute the loss using CrossEntropyLoss with ignore_index set to pad_token_id
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)  # Properly ignore padding tokens
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss
    
    training_args.save_strategy = "no"

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        compute_loss_func=custom_loss_fn,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_xla_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_xla_available()
        else None,
    )
                                                                                                                                         
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(os.path.join(training_args.output_dir,"final_model"))

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        trainer.compute_loss_func=None

        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        wandb.log({
            "perplexity": perplexity,
            "eval_accuracy": metrics["eval_accuracy"],
                   }, step=data_args.iteration)

    wandb.log({"data_selection_strategy": data_args.data_selection_strategy})

    if 'gt_cls_score' in train_dataset.features:
        human_samples = train_dataset.filter(
            lambda x: (x['gt_cls_score'] == 0),
            desc=f"Selecting only human samples",
        )
        ai_samples = train_dataset.filter(
            lambda x: (x['gt_cls_score'] == 1),
            desc=f"Selecting only human samples",
        )

        H_eff = len(human_samples)  # Effective human sample size
        A_eff = len(ai_samples)  # Effective AI sample size
        print(A_eff)
        print(H_eff)

        # wandb.log({"human_ai_ratio": H_eff/A_eff, "unique_human_samples": len(human_samples.unique("cls_text")), "unique_ai_samples": len(ai_samples.unique("cls_text"))})

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

if __name__ == "__main__":
    main()