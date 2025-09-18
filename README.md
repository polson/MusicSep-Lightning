# MusicSep-Lightning

A PyTorch Lightning toolkit for audio separation

## Features

- Live visualization of training progress
- Smart checkpoint loading with parameter matching
- Validation loss and SDR statistics while training
- Toolkit of `nn.Module`s for more easily building your own models

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

Both the `train` and `test` folders must contain audio files in WAV format corresponding to the stems you select in the
config's `target_sources` parameter.

## Usage

### Training

To start training the model:

```bash
python main.py
```

This will start training the model defined in `model/magsep/model.py` using the configuration from
`config/config.yaml`.

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

The inference will process audio files from the `inference_input` directory and separate them into the stems specified
in `target_sources`.

## Troubleshooting

### Progress Bar Not Displaying in PyCharm

If you're using PyCharm and the training progress bar is not showing up during training, enable the "Emulate terminal in
output console" option in your run configuration:

1. Go to **Run** → **Edit Configurations...**
2. Select your Python configuration
3. Under **Execution**, check the box for **"Emulate terminal in output console"**
4. Click **Apply** and **OK**

This will allow the progress bar to display correctly in PyCharm's output console.

## Suppressing Warnings

If you encounter verbose warning messages during training or inference, you can suppress them by using:

```bash
python -W ignore main.py
```

## License

MIT License - see [LICENSE](./LICENSE) for details.