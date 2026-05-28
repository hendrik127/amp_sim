# amp_sim

## Project structure
```
amp-sim/
│
├── data/
│   ├── input.wav    # DI guitar (dry input)
│   │── output.wav   # Amp recording (target)
│
├── runs/   # auto-generated experiment outputs 
├── src/
│   ├── train.py
│   ├── model.py
│   ├── dataset.py
│   ├── loss.py
│   │─── config.py
│
└── README.md
```

## Training data

Training data is included in git lfs. 

## Adding dependecies and creating environment

Have following tools installed:

- uv
- git lfs
- make

To install dependencies and intialize pre-commit hooks run
```
make install
```
add dependencies:
```
uv add <package-name>
```

## TODO
- test more architectures, make folder for models `src/models/`
- create pipeline to run multiple experiments with different architectures and hyperparameters, etc. Right now this can be done manually in `/src/config.py`
- Currently the model is fixed, iterate or find better architecture.

## Run training
```
uv run python src/train.py
```

## Training on HPC

```
make train_hpc
```

