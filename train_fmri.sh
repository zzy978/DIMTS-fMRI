#!/bin/bash
#SBATCH --partition=partition_1         # GPU 分区
#SBATCH --gres=gpu:1                       # 请求 1 块 GPU
#SBATCH --job-name=seq_256_zscore             # 作业名称
#SBATCH --ntasks=1                         # 任务数
#SBATCH --cpus-per-task=8                  # 每个任务分配的 CPU 核心数
#SBATCH --mem=32G                      # 分配的内存 (16GB)
#SBATCH --time=48:00:00   
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/%x_%j.err

python main.py --name fmri_seq256_pred128_zscore --config_file ./Config/fmri_seq256.yaml --gpu 0 --norm_method zscore --train