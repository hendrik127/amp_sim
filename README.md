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

install the [**uv**](https://docs.astral.sh/uv/) package manager
linux/macos
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
windows
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

to add dependency to project
```
uv add <package_name>
```

to add dev dependency to project
```
uv add --dev <package_name>
```

to install all the packages

```
uv sync
```

linting

```
uv run ruff check
```

formating

```
uv run ruff format
```


