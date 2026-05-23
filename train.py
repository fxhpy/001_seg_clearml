"""
DeepLabV3+ 分割训练
笔记本提交 → 5090 远程执行
"""

# ===== ClearML 必须在所有 import 之前 =====
import os, socket
socket.gethostname = lambda: "laptop"   # 防中文 hostname 出问题

from clearml import Task

# 显式指定依赖,避免 agent 自动分析失败
Task.force_requirements_env_freeze(requirements_file="requirements.txt")

task = Task.init(
    project_name='sealake/seg',
    task_name='deeplabv3plus-mobilenet-remote-v1',
)

# 所有可调参数集中放,Web UI 上能看能改
config = task.connect({
    "train_root": r"C:\Users\admin\Desktop\Unet\dataset\data2026\Train",
    "val_root":   r"C:\Users\admin\Desktop\Unet\dataset\data2026\Val",
    "batch_size": 16,
    "num_epochs": 100,
    "max_lr": 1e-3,
    "weight_decay": 1e-4,
    "num_classes": 4,
    "encoder": "mobilenet_v2",
    "checkpoint_dir": "checkpoints_20260508",
})

# ⭐ 关键这行:把任务推到 5090 的 default 队列
task.execute_remotely(queue_name="default")

# ===== 以下代码在 5090 上执行 =====

import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from dataset import DroneDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.makedirs(config["checkpoint_dir"], exist_ok=True)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("data load")
    train_set = DroneDataset(config["train_root"], True)
    val_set = DroneDataset(config["val_root"], False)
    print(f"Train: {len(train_set)} images, Val: {len(val_set)} images")

    train_loader = DataLoader(
        train_set, batch_size=config["batch_size"],
        shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_set, batch_size=config["batch_size"],
        shuffle=False, num_workers=0,
    )

    print("model load")
    model = smp.DeepLabV3Plus(
        config["encoder"],
        encoder_weights='imagenet',
        classes=config["num_classes"],
        activation=None,
        encoder_depth=5,
        decoder_channels=256,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["max_lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, config["max_lr"],
        epochs=config["num_epochs"],
        steps_per_epoch=len(train_loader),
    )

    writer = SummaryWriter()
    logger = task.get_logger()

    print("train start")
    for epoch in range(config["num_epochs"]):
        train_loss = 0
        val_loss = 0

        # train
        model.train()
        iteration_train = 0
        for i, (img, mask) in enumerate(train_loader):
            img = img.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            iteration_train += 1
            print(f'Epoch:{epoch}, Iter:{i}, Loss:{loss.item():.4f}')

        # val
        model.eval()
        iteration_val = 0
        for i, (imgs, labels) in enumerate(val_loader):
            with torch.no_grad():
                imgs = imgs.to(device)
                labels = labels.to(device)
                output = model(imgs)
                loss = criterion(output, labels)
                val_loss += loss.item()
                iteration_val += 1

        train_loss_avg = train_loss / iteration_train
        val_loss_avg   = val_loss / iteration_val
        print(f"=== Epoch {epoch}: train={train_loss_avg:.4f} val={val_loss_avg:.4f} ===")

        writer.add_scalar("train_loss", train_loss_avg, epoch)
        writer.add_scalar("val_loss",   val_loss_avg,   epoch)
        logger.report_scalar("loss", "train", train_loss_avg, epoch)
        logger.report_scalar("loss", "val",   val_loss_avg,   epoch)

        ckpt_path = os.path.join(
            config["checkpoint_dir"],
            f'ep{epoch+1:03d}-loss{train_loss_avg:.3f}-val_loss{val_loss_avg:.3f}.pth'
        )
        torch.save(model.state_dict(), ckpt_path)

    print("Done!")