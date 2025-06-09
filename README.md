# Rat_Gift_ICA

This repository is part of the final project for the BME course "Python Programming" at ShanghaiTech University. The main objective is to use machine learning to separate rat vocal signals from background noise.

The project is built on top of [PyTorch](https://pytorch.org/) and [scikit-learn](https://scikit-learn.org/).

> ⚠️ This project is under active development and may contain bugs.

---

## Preparation

### 1. Install Conda (Virtual Environment Manager)
If you do not have Conda installed, please refer to the [official documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

---

## Project Requirements

### 1. Project Dependencies
Create and activate a virtual environment, then install the required packages:

```bash
# Create environment
conda create -n ica python=3.11

# Activate environment
conda activate ica

# Install packages
pip install -r requirements.txt
```

> **Note:** The `requirements.txt` may contain more packages than strictly necessary. You can also check each file's dependencies and install only what you need, as this project is relatively lightweight.

### 2. Accelerate Configuration
The project uses [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index) for efficient training. You can find the configuration file in `configs/accelerate_config.yaml`.

Adjust the config file to fit your hardware settings.

### 3. Datasets
The datasets used in this project can be downloaded from: [https://epan.shanghaitech.edu.cn/l/hFkGi2](https://epan.shanghaitech.edu.cn/l/hFkGi2)

Place the downloaded data in the `data/` directory.

---

## How to Reproduce Training Results

1. Choose the model configuration you want to use for training. All model configs are in the `configs/` directory.
2. Start training with the following command:

```bash
python run_train.py --config configs/resnet18.yaml
```

- For scikit-learn models (KNN, Logistic Regression, etc.), use `sk_pred.ipynb`.
- To visualize the dataset and training results, refer to `data_player.ipynb` and `draw_results.ipynb`.

---

## For Developers

### Project Structure
Below are the main files and their descriptions:

| File                   | Description                                                      |
|------------------------|------------------------------------------------------------------|
| `data_player.ipynb`    | Interactive notebook for data visualization and exploration.      |
| `sk_pred.ipynb`        | Notebook for training and evaluating scikit-learn models.         |
| `deep_models.py`       | Implementation of deep learning model architectures.              |
| `data_prepare.py`      | Data loading, preprocessing, and dataset utilities.               |
| `deep_training.py`     | Training and evaluation logic for deep learning models.           |
| `run_train.py`         | Main script to launch training with configuration support.        |
| `draw_results.ipynb`   | Notebook for plotting and analyzing training results.             |
| `requirements.txt`     | Python dependencies for the project.                             |
| `configs/`             | Model and training configuration files (YAML format).             |
| `data/`                | Directory for storing datasets (not included in repo).            |

---

## Contact
For questions or suggestions, please contact anyone of the team members.