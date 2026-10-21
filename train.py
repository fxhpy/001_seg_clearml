from clearml import Task

Task.force_requirements_env_freeze(requirements_file="requirements.txt")

task = Task.init(
    project_name="sealake-seg",
    task_name="train"
)

task.execute_remotely(queue_name="default", exit_process=True)
config = {
    # ===== 数据来源：通过引用最近的数据下载任务 =====
    # 留空 → 自动找最近一次成功的 download-lakefs-data 任务
    # 填具体 task_id → 用指定的数据版本
    "data_task_id": "",

    # 训练超参
    "batch_size": 16,
    "max_lr": 1e-3,
    "epoch": 100,  # 先跑通改 5
    "weight_decay": 1e-4,
    "num_workers": 4,

    # 模型
    "encoder": "mobilenet_v2",
    "encoder_weights": "imagenet",
    "num_classes": 4,
    "encoder_depth": 5,
    "decoder_channels": 256,

    "checkpoint_dir": "checkpoints",
}
config = task.connect(config)

task.execute_remotely(queue_name="default", exit_process=True)

# ===== 下面在 5090 Agent 上执行 =====
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import DroneDataset


def get_data_paths():
    """通过 ClearML 查找数据下载任务，拿到本地路径"""
    if config["data_task_id"]:
        # 指定了具体的数据任务 ID
        data_task = Task.get_task(task_id=config["data_task_id"])
        print(f"Using specified data task: {config['data_task_id']}")
    else:
        # 自动找最近一次成功的数据下载任务
        data_task = Task.get_task(
            project_name='seg-clearml',
            task_name='download-lakefs-data',
            task_filter={'status': ['completed']},
        )
        if data_task is None:
            raise RuntimeError(
                "No completed data download task found. "
                "Please run download_data.py first."
            )
        print(f"Found latest data task: {data_task.id}")

    # 从 task 的 user_properties 取本地路径
    props = data_task.get_user_properties()
    local_root = props.get('local_data_root', {}).get('value')
    lakefs_ref = props.get('lakefs_ref', {}).get('value')

    if not local_root:
        # fallback：从 artifact 取
        info = data_task.artifacts['data_info'].get()
        local_root = info['local_path']
        lakefs_ref = info['lakefs_ref']

    print(f"Data root: {local_root}")
    print(f"LakeFS ref: {lakefs_ref}")

    # 记录到当前训练任务，方便追溯
    task.set_user_properties(
        used_data_task=data_task.id,
        lakefs_ref=lakefs_ref,
    )

    return local_root


def main():
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
        print(f"capability: {torch.cuda.get_device_capability(0)}")

    # ===== 关键：动态获取数据路径 =====
    data_root = get_data_paths()
    train_root = os.path.join(data_root, "Train")
    val_root = os.path.join(data_root, "Val")

    print(f"train_root: {train_root}")
    print(f"val_root:   {val_root}")

    print("data load")
    train_set = DroneDataset(train_root, True)
    val_set = DroneDataset(val_root, False)

    train_loader = DataLoader(
        train_set, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"],
    )
    val_loader = DataLoader(
        val_set, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"],
    )

    print("model load")
    model = smp.DeepLabV3Plus(
        config["encoder"], encoder_weights=config["encoder_weights"],
        classes=config["num_classes"], activation=None,
        encoder_depth=config["encoder_depth"],
        decoder_channels=config["decoder_channels"],
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["max_lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, config["max_lr"],
        epochs=config["epoch"], steps_per_epoch=len(train_loader),
    )

    writer = SummaryWriter()

    print("train start")
    for epoch in range(config["epoch"]):
        train_loss = 0
        iteration_train = 0
        model.train()
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
            if i % 10 == 0:
                print(f'Epoch:{epoch}, Iter:{i}, Loss:{loss.item():.4f}')

        val_loss = 0
        iteration_val = 0
        model.eval()
        for i, (imgs, labels) in enumerate(val_loader):
            with torch.no_grad():
                imgs = imgs.to(device)
                labels = labels.to(device)
                output = model(imgs)
                loss = criterion(output, labels)
                val_loss += loss.item()
                iteration_val += 1

        avg_train_loss = train_loss / iteration_train
        avg_val_loss = val_loss / iteration_val
        writer.add_scalar("train_loss", avg_train_loss, epoch)
        writer.add_scalar("val_loss", avg_val_loss, epoch)
        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        ckpt_path = os.path.join(
            config["checkpoint_dir"],
            f'ep{epoch+1:03d}-loss{avg_train_loss:.3f}-val_loss{avg_val_loss:.3f}.pth'
        )
        torch.save(model.state_dict(), ckpt_path)

if __name__ == "__main__":
    main()
