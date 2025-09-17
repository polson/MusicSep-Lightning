# MusicSep-Lightning
A complete PyTorch Lightning toolkit for separating musical stems from audio tracks. This project provides an end-to-end
solution for music source separation using deep learning techniques.

## Features
- PyTorch Lightning integration for scalable training
- Live visualization of training progress
- Toolkit of `nn.Module`s for more easily building your own models
- Sample model in ./model/magsep/model.py

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

## Dataset Setup
The dataset directory you specify in the config must have the following structure:
```
dataset/
├── train/
└── test/
```

Both the `train` and `test` folders must contain audio files in **WAV format** corresponding to the stems you select in the config's `target_sources` parameter.

## Usage

### Training
To start training the model:
```bash
python main.py
```
This will start training the model defined in `./model/magsep/model.py` using the configuration from
`./config/config.yaml`.

### Inference
To run inference on audio files:

1. **Set the mode to inference** in `./config/config.yaml`:
   ```yaml
   mode: inference
   ```

2. **Specify the input folder** containing your audio files:
   ```yaml
   paths:
     inference_input: "/path/to/your/audio/files"
   ```

3. **Provide a trained checkpoint** to load:
   ```yaml
   paths:
     resume_from: "/path/to/your/checkpoint.ckpt"
   ```

4. **Run inference**:
   ```bash
   python main.py
   ```

The inference will process audio files from the `inference_input` directory and separate them into the stems specified in `target_sources`.

## License
MIT License - see [LICENSE](./LICENSE) for details.
