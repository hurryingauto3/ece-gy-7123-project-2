# Parameter‑Efficient RoBERTa Fine‑Tuning on AGNews with LoRA

## Overview  
Efficiently adapt RoBERTa‑base (125 M params) for AGNews (120 k train, 7.6 k test, 4 classes) within a 1 M‑parameter budget using Low‑Rank Adaptation (LoRA). Systematic experiments vary adapter **rank** (_r_), **scaling** (α), **modules** (attention QKV, FFN, classifier), **data strategies** (augmentation, filtering) and **regularization** (dropout, label smoothing).  

Key result:  
- **94.61 %** validation accuracy (loss 0.1704) with ~962 k trainable params  
- **0.8420** hidden‑test score on Kaggle  

## Repository structure  
    ├─ code/
    │  ├─ dl-project-2-4-21-1130.ipynb      # end‑to‑end training & evaluation
    │  ├─ results_reproduction.ipynb        # reproduce best result
    │  ├─ requirements.txt                  # Python dependencies
    │  └─ results/                          # per‑experiment logs, checkpoints, plots
    └─ report/
        ├─ main.tex                          # AAAI paper source
        └─ images/                           # figures for report

## Setup  
1. Clone and create environment  
   ```bash
   git clone https://github.com/hurryingauto3/ece-gy-7143-project-2.git
   cd ece-gy-7143-deeplearning
   conda create -n lora-agnews python=3.10
   conda activate lora-agnews
   pip install -r requirements.txt

	2.	Download AGNews via HuggingFace Datasets (handled in notebook).

Usage
	1.	Open and run dl-project-2-4-21-1130.ipynb
	2.	Adjust LoRAConfig(rank, alpha, target_modules)
	3.	Launch training:

trainer.train()
trainer.evaluate()
trainer.predict(test_dataset)


	4.	Artifacts saved under results/<exp_id>/.

Results & Artifacts
	•	Checkpoints: results/<exp_id>/results/checkpoint-*/adapter_model.safetensors
	•	Metrics: eval_results.json, submission_acc_*.csv
	•	Plots: combined_plots.png, training/validation curves
	•	Adapters: results/<exp_id>/trained_adapters/

Reproduce & Extend
	•	Edit hyperparameters in the “# CONFIGURE EXPERIMENT” cell
	•	Modify Trainer args for optimizers/schedulers
	•	Add augmentation/regularization in data pipeline

Citation

@article{hu2022lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={arXiv preprint arXiv:2106.09685},
  year={2022},
}

