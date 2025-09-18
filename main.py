import inspect
import logging
import os
import random
from datetime import datetime

import lightning as L
import numpy as np
import torch
from lightning.fabric import disable_possible_user_warnings
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from callback.progress_bar import AudioSeparationProgressBar
from config.config_loader import get_config
from util.checkpoint_loader import SmartCheckpointLoader
from trainer import AudioSourceSeparation


def create_model(c):
    return AudioSourceSeparation(
        config=c
    )


def load_model(config):
    print(f"Loading model with smart checkpoint loader from: {config.paths.resume_from}")
    loader = SmartCheckpointLoader()
    try:
        model = AudioSourceSeparation(config)
        loader.load_checkpoint(model, config.paths.resume_from)
        return model
    except Exception as e:
        print(f"Smart loading failed: {e}")
        print("Falling back to PyTorch Lightning loader...")
        return AudioSourceSeparation.load_from_checkpoint(
            config.checkpoint_path,
            config=config,
            strict=False
        )


def setup_logger(config):
    logger = False
    if config.logging.wandb.enabled:
        model_name = model.model.__class__.__name__
        now = datetime.now()
        formatted = now.strftime('%Y-%m-%d %H:%M:%S')
        logger = WandbLogger(
            name=model_name + " - " + formatted,
            log_model=False,
            config=config,
        )

        if config.logging.wandb.save_code:
            model_path = os.path.dirname(inspect.getfile(model.model.__class__))
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    logger.experiment.save(file_path, base_path=model_path)
    return logger


def create_trainer():
    os.makedirs(config.paths.checkpoints, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.paths.checkpoints,
        filename="audio-separation-{epoch:02d}-{val_avg_sdr:.4f}",
        save_top_k=-1,
        save_on_train_epoch_end=False,
        save_last=True,
        monitor="val_avg_sdr",
        mode="max",
        every_n_train_steps=None
    )

    callbacks = [
        AudioSeparationProgressBar(target_sources=config.training.target_sources),
        checkpoint_callback,
    ]

    return L.Trainer(
        limit_train_batches=config.training.num_steps,
        max_epochs=config.training.max_epochs,
        accelerator="auto",
        callbacks=callbacks,
        logger=setup_logger(config),
        precision=config.training.precision,
        log_every_n_steps=1,
        detect_anomaly=False,
        accumulate_grad_batches=config.training.grad_accum,
        num_sanity_val_steps=0,
        gradient_clip_val=config.training.grad_clip,
        limit_val_batches=config.validation_steps
    )


def setup_logging():
    logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)
    disable_possible_user_warnings()


def run_inference(trainer, model, config):
    print(f"\nRunning inference on: {config.paths.inference_input}")
    results = trainer.predict(model=model)
    print(f"Inference completed! Processed {len(results)} files.")
    print(f"Results saved to: {config.paths.validation}")
    return results


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    L.seed_everything(seed, workers=True)


def initialize(config):
    if config.training.seed is not None:
        set_seed(config.training.seed)
    torch.cuda.set_per_process_memory_fraction(0.9)
    torch.set_float32_matmul_precision("medium")


if __name__ == "__main__":
    config = get_config("config/config.yaml")
    initialize(config)
    if config.paths.resume_from:
        try:
            model = load_model(config)
        except Exception as e:
            print(f"PyTorch Lightning loader also failed: {e}")
            print("Creating new model...")
            model = create_model(config)
    else:
        model = create_model(config)

    trainer = create_trainer()

    if config.mode == "inference":
        run_inference(trainer, model, config)
        exit(0)

    print("\nStarting training for", config.training.max_epochs, "epochs...")
    trainer.fit(
        model=model
    )
    print(f"Completed {trainer.current_epoch}/{config.training.max_epochs} epochs!")
