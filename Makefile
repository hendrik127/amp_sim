install:
	uv sync
	uv run pre-commit install

train_hpc:
	sbatch job.sh