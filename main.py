import hydra
from omegaconf import DictConfig
import os
import subprocess
from hydra.core.hydra_config import HydraConfig, OmegaConf
import json
from src.utils.utils import plot_metric_over_generations
import wandb

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # wandb parameters
    os.environ["WANDB_DISABLED"] = str(cfg.wandb_disabled)
    os.environ["WANDB_RUN_GROUP"] = "experiment-" + wandb.util.generate_id()
    wandb_project = f"model_collapse_{cfg.model.short_name}"
    if cfg.accumulate_ai_data:
        prepend = "accumulate_"
        print(cfg.data_selection.strategy)
        if cfg.data_selection.strategy != 'None':
            prepend += "select_"
        wandb_project = prepend + wandb_project
    os.environ["WANDB_PROJECT"] = wandb_project
    
    # add config parameters
    if not bool(str(cfg.wandb_disabled)):
        wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.finish()

    # cuda device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.cuda_device)

    # get variables from configs
    experiment_path = HydraConfig.get().run.dir
    model_name = cfg.model.name
    block_size = cfg.train.block_size
    loss_on_last_n_tokens = cfg.train.loss_on_last_n_tokens
    batch_size = cfg.train.batch_size
    seed = cfg.seed
    num_train_epochs = cfg.train.num_train_epochs
    num_iterations = cfg.num_iterations
    num_samples = cfg.train.num_samples
    low_cpu_mem_usage = str(cfg.low_cpu_mem_usage)
    torch_dtype = str(cfg.torch_dtype)

    if cfg.smoke_test:
        num_samples = "100"
        num_iterations = 1
        low_cpu_mem_usage = True
        torch_dtype = "bfloat16"

    # Initial training
    subprocess.run([
        "python", "src/train.py",
        "--model_name_or_path", model_name,
        "--output_dir", f"{experiment_path}/0/model/",
        "--per_device_train_batch_size", str(batch_size),
        "--per_device_eval_batch_size", str(batch_size),
        "--train_file", f"{cfg.dataset.path}/train.json",
        "--test_file", f"{cfg.dataset.path}/test.json",
        "--do_train",
        "--do_eval",
        "--ai_beta", str(cfg.ai_beta),
        "--block_size", str(block_size),
        "--loss_on_last_n_tokens", str(loss_on_last_n_tokens),
        "--num_train_epochs", str(num_train_epochs),
        "--save_steps", str(cfg.train.save_steps),
        "--seed", str(seed),
        "--torch_dtype", torch_dtype,
        "--low_cpu_mem_usage", low_cpu_mem_usage,
        "--run_name", "train_iteration_0",
        "--iteration", "0",
        "--preprocessing_num_workers", "1"
    ])

    # Iterative train-generate cycles
    for iteration in range(1, num_iterations+1):

        prev_model_output_dir = f"{experiment_path}/{iteration - 1}/model"

        command = [
            "python", "src/generate.py",
            "--model_name", model_name,
            "--model_path", os.path.join(prev_model_output_dir, "final_model"),
            "--input_token_length", str(block_size - loss_on_last_n_tokens),
            "--block_size", str(block_size),
            "--iteration", str(iteration),
            "--experiment_path", str(experiment_path),
            "--dataset_filepath", f"{cfg.dataset.path}/train.json",
            "--seed", str(seed),
            "--temperature", str(cfg.decoding.temperature),
            "--top_p", str(cfg.decoding.top_p),
            "--top_k", str(cfg.decoding.top_k),
            "--torch_dtype", torch_dtype,
            "--low_cpu_mem_usage", low_cpu_mem_usage,
            "--classify_text", "1",
            "--detector_tokenizer_name", str(cfg.detector.tokenizer_name),
            "--detector_path", str(cfg.detector.model_path),
            "--detector_threshold", str(cfg.detector.ai_confidence_threshold),
            "--detector_temperature", str(cfg.detector.temperature),
        ]
        if num_samples != "None":
            command.extend(["--num_samples", num_samples])

        subprocess.run(command)
    
        train_file = f"{experiment_path}/{iteration}/data.json"
        new_model_output_dir = f"{experiment_path}/{iteration}/model"

        subprocess.run([
            "python", "src/train.py",
            "--model_name_or_path", model_name,
            "--output_dir", new_model_output_dir,
            "--per_device_train_batch_size", str(batch_size),
            "--per_device_eval_batch_size", str(batch_size),
            "--train_file", train_file,
            "--test_file", f"{cfg.dataset.path}/test.json",
            "--experiment_path", experiment_path,
            "--human_data_filepath", f"{cfg.dataset.path}/train.json",
            "--human_data_alpha", str(cfg.human_data_alpha),
            "--ai_beta", str(cfg.ai_beta),
            "--do_train",
            "--do_eval",
            "--block_size", str(block_size),
            "--loss_on_last_n_tokens", str(loss_on_last_n_tokens),
            "--num_train_epochs", str(num_train_epochs),
            "--save_steps", str(cfg.train.save_steps),
            "--seed", str(seed),
            "--torch_dtype", torch_dtype,
            "--low_cpu_mem_usage", low_cpu_mem_usage,
            "--run_name", f"train_iteration_{iteration}",
            "--iteration", str(iteration),
            "--data_selection_strategy", str(cfg.data_selection.strategy),
            "--accumulate_ai_data", str(cfg.accumulate_ai_data),
            "--preprocessing_num_workers", "2",
            "--upsample_factor", str(cfg.data_selection.upsample_factor),
            "--bias_factor", str(cfg.data_selection.bias_factor),
            "--max_repeats", str(cfg.data_selection.max_repeats)
        ])

    if cfg.plotting.enabled:
        eval_results = {}
        
        iterations = [str(i) for i in range(num_iterations)]

        for iteration in iterations:
            file_path = f"{experiment_path}/{iteration}/model/eval_results.json"
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    eval_results[int(iteration)] = json.load(f)
            else:
                print(f"File not found: {file_path}")
        
        if not os.path.exists(f"{experiment_path}/plots"):
            os.makedirs(f"{experiment_path}/plots")

        for metric in cfg.plotting.metrics:
            plot_metric_over_generations(eval_results=eval_results, metric_name=metric, y_label=metric.capitalize(), label=f'Block Size {block_size}', color='blue', save_path=f"{experiment_path}/plots/{metric}.png")

if __name__ == "__main__":
    main()
    wandb.finish()