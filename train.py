"""
DeepLabV3+ on sealake dataset
通过 ClearML 提交到 5090 训练
"""

# ===== ClearML 必须最先初始化 =====
import os, socket
socket.gethostname = lambda: "laptop"

from clearml import Task

Task.force_requirements_env_freeze(requirements_file="requirements.txt")

task = Task.init(
    project_name="sealake/seg",
    task_name="deeplabv3plus-mobilenet-debug",
)

# 所有可调参数集中在这里
config = task.connect({
    # 数据路径(5090 上的实际路径)
    "train_root": r"D:\data\sealake\Train",
    "val_root":   r"D:\data\sealake\Val",
    # 超参
    "batch_size": 4,           # 先小,跑通再加大
    "num_epochs": 2,           # 先 2 epoch,跑通再加
    "max_lr": 1e-3,
    "weight_decay": 1e-4,
    "num_classes": 4,
    # 模型
    "encoder": "mobilenet_v2",
    "encoder_weights": "imagenet",
    "checkpoint_dir": "checkpoints",
})

# 推到 5090 队列
task.execute_remotely(queue_name="default")

# ===== 以下代码在 5090 上执行 =====

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import DroneDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("Loading data...")
train_set = DroneDataset(config["train_root"], need_handle=True)
val_set = DroneDataset(config["val_root"], need_handle=False)
print(f"Train: {len(train_set)} images")
print(f"Val:   {len(val_set)} images")

train_loader = DataLoader(
    train_set, batch_size=config["batch_size"],
    shuffle=True, num_workers=0,
)
val_loader = DataLoader(
    val_set, batch_size=config["batch_size"],
    shuffle=False, num_workers=0,
)

print("Building model...")
model = smp.DeepLabV3Plus(
    encoder_name=config["encoder"],
    encoder_weights=config["encoder_weights"],
    classes=config["num_classes"],
    activation=None,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["max_lr"],
    weight_decay=config["weight_decay"],
)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=config["max_lr"],
    epochs=config["num_epochs"],
    steps_per_epoch=len(train_loader),
)

writer = SummaryWriter()
os.makedirs(config["checkpoint_dir"], exist_ok=True)
logger = task.get_logger()

print("Training starts...")
for epoch in range(config["num_epochs"]):
    # Train
    model.train()
    train_loss = 0
    for i, (img, mask) in enumerate(train_loader):
        img = img.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, mask)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
        if i % 5 == 0:
            print(f"Epoch {epoch} Iter {i} Loss {loss.item():.4f}")
            logger.report_scalar("loss_iter", "train", loss.item(),
                                 epoch * len(train_loader) + i)

    # Val
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for img, mask in val_loader:
            img = img.to(device)
            mask = mask.to(device)
            out = model(img)
            val_loss += criterion(out, mask).item()

    train_avg = train_loss / max(len(train_loader), 1)
    val_avg = val_loss / max(len(val_loader), 1)
    print(f"=== Epoch {epoch}: train={train_avg:.4f}, val={val_avg:.4f} ===")

    writer.add_scalar("train_loss", train_avg, epoch)
    writer.add_scalar("val_loss", val_avg, epoch)
    logger.report_scalar("loss_epoch", "train", train_avg, epoch)
    logger.report_scalar("loss_epoch", "val", val_avg, epoch)

    ckpt_path = os.path.join(
        config["checkpoint_dir"],
        f"ep{epoch+1:03d}-train{train_avg:.3f}-val{val_avg:.3f}.pth"
    )
    torch.save(model.state_dict(), ckpt_path)

print("Done!")