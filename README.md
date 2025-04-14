# Machine-Generated Text Detection Prevents Language Model Collapse

[![arXiv](https://img.shields.io/badge/arXiv-2502.15654-b31b1b.svg)](https://arxiv.org/abs/2502.15654)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository contains the official implementation of the paper ["Machine-generated text detection prevents language model collapse"](https://arxiv.org/abs/2502.15654) by George Drayson, Emine Yilmaz, and Vasileios Lampos.

## Overview

As Large Language Models (LLMs) become increasingly prevalent, their generated outputs are proliferating across the web, risking a future where machine-generated content dilutes human-authored text. This project investigates the impact of decoding strategy on model collapse and proposes an importance sampling approach to alleviate model collapse.

## Repository Structure

```
.
├── config/              # Configuration files
├── data/               # Dataset directory
├── src/               # Source code
│   ├── train.py       # Training script
│   ├── generate.py    # Generation script
│   ├── load_data.py   # Load all dataset files
│   └── utils/         # Utility functions
├── requirements.txt    # Python dependencies
└── main.py            # Main training loop
```

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

2. Install transformers from source:
```bash
pip install git+https://github.com/huggingface/transformers
```

3. Install other requirements:
```bash
pip install -r requirements.txt
```

## Quickstart

1. Load and prepare the dataset:
```bash
python src/load_data.py
```

2. To start the recursive training process:
```bash
python main.py
```

The training process can be customised using different configuration files (see Configuration section below).



## Configuration

The project uses [Hydra](https://hydra.cc/) for configuration management. Key configuration files are located in the `config/` directory:
- `config.yaml`: Main configuration file
- `decoding/`: Decoding-specific configurations
- `model/`: Model-specific configurations
- `detector/`: Detector-specific configurations
- `train/`: Training hyperparameters

### Running with Different Configurations

You can override any configuration parameter from the command line:
```bash
# Change training parameters
python main.py train.batch_size=16 train.num_train_epochs=5

# Modify decoding strategy
python main.py decoding=beam_search

# Use a different detector
python main.py detector.model_path=detectors/custom_detector
```

For more details on configuration options, see the [Hydra documentation](https://hydra.cc/docs/intro/).

## Model

The trained model is available on the Hugging Face Hub at [GeorgeDrayson/modernbert-ai-detection](https://huggingface.co/GeorgeDrayson/modernbert-ai-detection). It is a fine-tuned version of ModernBERT-base trained on the MAGE dataset for machine-generated text detection.

### Model Details
- Model Size: 150M parameters
- Base Model: answerdotai/ModernBERT-base
- Dataset: yaful/MAGE
- Task: Text Classification

## Experiment Tracking
The project uses Weights & Biases for experiment tracking. Results, metrics, and artifacts are automatically logged during training. To view your results set your API key: `export WANDB_API_KEY=your_key_here`

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about the code or paper, please open an issue in this repository.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{drayson2025machine,
  title={Machine-generated text detection prevents language model collapse},
  author={Drayson, George and Yilmaz, Emine and Lampos, Vasileios},
  journal={arXiv preprint arXiv:2502.15654},
  year={2025}
}
```