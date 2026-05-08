import timm
import torch
from tqdm.auto import tqdm
from typing import Optional, Tuple, Dict, List
import matplotlib.pyplot as plt
import os


def load_pretrained_vit_checkpoint(model, device):
    """Load pretrained ViT-Base weights from timm into our custom ViT model"""
    # load timm ViT-base pretrained
    timm_model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    timm_model = timm_model.to(device)
    timm_state = timm_model.state_dict()

    # Map & Load weights
    custom_state = model.vit.state_dict()
    pretrained_keys = 0

    for key in custom_state.keys():
        if key in timm_state:
            custom_state[key] = timm_state[key]
            pretrained_keys += 1

    model.vit.load_state_dict(custom_state, strict=False)

    print(f"Loaded pretrained ViT-Base ckpt from timm")
    print(f"    Keys loaded: {pretrained_keys}/{len(custom_state)}")
    print(f"    Status: Ready for fine-tuning")

    return model



def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating....")
        for videos, labels in pbar:
            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            loss = criterion(outputs, labels)

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



def denormalize(frames):
    """Denormalize for visualizing"""
    frames = frames.clone()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1) # mean & std of ImageNet
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
    frames = frames * std + mean
    return frames.clamp(0,1)


def save_plot_from_history(history: Dict[str, List], save_dir: str, plot_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(15,6))

    # plot loss
    axes[0].plot(history['train_loss'], label='Train', marker = 'o')
    axes[0].plot(history['val_loss'], label='Val', marker='s')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # plot accuracy
    axes[1].plot(history['train_acc'], label='Train', marker = 'o')
    axes[1].plot(history['val_acc'], label='Val', marker='s')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim([0,1])

    plt.tight_layout()
    plot_path = file_path = os.path.join(save_dir, plot_name)
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print("Plot saved!")
