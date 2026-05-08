from __future__ import annotations
import math
import random
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import re
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from tqdm.auto import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import timm

from dataset import HMDB51Dataset, collate_fn
from model import BasicViT, ViTGRU, ViTLSTM
from utils import load_pretrained_vit_checkpoint, evaluate, save_plot_from_history

from config import (
    DATA_ROOT,
    OUTPUT_DIR,
    VAL_RATIO,
    TEST_RATIO,
    SEED,
    NUM_FRAMES,
    FRAME_STRIDE,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """

    Return:
        epoch_loss: float, epoch_acc: float
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training...")
    for videos, labels in pbar:
        videos = videos.to(device)
        labels = labels.to(device)

        outputs = model(videos)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': f'{total_loss / (pbar.n + 1):.4f}',
                         'acc': f'{correct / total:.4f}'})

    epoch_loss = total_loss / len(dataloader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

def main():
    train_dataset = HMDB51Dataset(
        root=DATA_ROOT,
        split="train",
        num_frames=NUM_FRAMES,
        frame_stride=FRAME_STRIDE,
        image_size=IMG_SIZE,
        val_ratio=VAL_RATIO,
        seed=SEED,
    )

    val_dataset = HMDB51Dataset(
        root=DATA_ROOT,
        split="val",
        num_frames=NUM_FRAMES,
        frame_stride=FRAME_STRIDE,
        image_size=IMG_SIZE,
        val_ratio=VAL_RATIO,
        seed=SEED,
    )

    test_dataset = HMDB51Dataset(
        root=DATA_ROOT,
        split="test",
        num_frames=NUM_FRAMES,
        frame_stride=FRAME_STRIDE,
        image_size=IMG_SIZE,
        val_ratio=VAL_RATIO,
        seed=SEED,
    )

    print(f"Train clips: {len(train_dataset)}")
    print(f"Train Class count: {len(train_dataset.classes)}")

    print(f"Val clips: {len(val_dataset)}")
    print(f"Val Class count: {len(val_dataset.classes)}")

    print(f"Test clips: {len(test_dataset)}")
    print(f"Test Class count: {len(test_dataset.classes)}")


    train_loader = DataLoader(train_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=NUM_WORKERS,
                            collate_fn=collate_fn)

    val_loader = DataLoader(val_dataset,
                            BATCH_SIZE,
                            shuffle=False,
                            num_workers=NUM_WORKERS,
                            collate_fn=collate_fn)

    test_loader = DataLoader(test_dataset,
                            BATCH_SIZE,
                            shuffle=False,
                            num_workers=NUM_WORKERS,
                            collate_fn=collate_fn)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")


    # initialize model
    model = ViTGRU(num_classes=len(train_dataset.classes)).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"   Model created")
    print(f"   Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"   Architecture: ViT-Base from scratch")
    print(f"   Status: Untrained (from scratch)")


    # load pretrained weights
    model = load_pretrained_vit_checkpoint(model, DEVICE)


    # training setup
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=LEARNING_RATE,
                                weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Training setup:")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Trainable parameters: {trainable_params:,} (ALL params)")
    print(f"   Optimizer: Adam")
    print(f"   Scheduler: CosineAnnealing")


    history = {'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'train_time': []}
    best_acc = 0.0
    best_model_path = os.path.join(OUTPUT_DIR,'vit_gru_best.pt')

    print(f"\n{'='*63}")
    print(f"Fine-tuning ViT w GRU for {EPOCHS} epochs")
    print(f"\n{'='*63}")


    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        duration = time.time() - start

        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)


        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['train_time'].append(duration)

        print(f"  -Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"  -Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"    Best model saved with acc = {best_acc:.4f}")

        scheduler.step()

    print(f"\n{'='*63}")
    print(f"Training complete!")
    print(f"Best val accuracy: {best_acc:.4f}")
    print(f"\n{'='*63}")


    save_plot_from_history(history, OUTPUT_DIR, "vit_gru.png")

if __name__ == "__main__":
    main()
