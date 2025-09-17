# MusicSep-Lightning

A complete PyTorch Lightning toolkit for separating musical stems from audio tracks. This project provides an end-to-end
solution for music source separation using deep learning techniques.

## Features

- PyTorch Lightning integration for scalable training

## Requirements

- Python 3.8+
- CUDA-compatible GPU

## Installation

### 1. Create Virtual Environment

First, create and activate a Python virtual environment:

```bash
# Create virtual environment
python -m venv musicsep-env

# Activate virtual environment
# On Windows:
musicsep-env\Scripts\activate
# On macOS/Linux:
source musicsep-env/bin/activate
```

### 2. Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

## Usage

To start training the model:

```bash
python main.py
```

This will start training the model defined in `./models/magsep/model.py` using the configuration from
`./config/config.yaml`.

## License

MIT License - see [LICENSE](./LICENSE) for details.
