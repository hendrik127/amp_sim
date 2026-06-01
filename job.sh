#!/bin/bash
#SBATCH --job-name=seg_train          # Jobi nimi
#SBATCH --account=ealloc_ati-1-neur     # Sinu ETAIS konto
#SBATCH --partition=gpu                # Partitsioon (GPU jaoks)
#SBATCH --gres=gpu:a100-40g:1                   # GPU arv, kui vaja mitu muuta
#SBATCH --cpus-per-task=4              # CPU tuumade arv
#SBATCH --mem=32G                      # Mälumaht
#SBATCH --time=30:00:00                # Maksimaalne tööaeg (HH:MM:SS)
#SBATCH --output=logs/slurm-%j.out          # Logi fail
# Keskkonna seadistamine
# -------------------------

# Liigu projekti kataloogi
cd ~/repos/amp_sim/

uv sync

# Käivita treening
uv run python src/train.py



