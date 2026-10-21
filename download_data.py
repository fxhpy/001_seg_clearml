import os
from clearml import Task

# ===== ClearML 任务 =====
Task.force_requirements_env_freeze(requirements_file="requirements.txt")

task = Task.init(
    project_name='seg-clearml',
    task_name='download-lakefs-data',
    task_type=Task.TaskTypes.data_processing,  # 标记为数据处理任务
)

# ===== 数据下载配置（可在 ClearML UI 修改）=====
config = {
    # LakeFS 连接
    "lakefs_endpoint": "http://202.120.42.71:8000",

    # 数据集位置
    "lakefs_repo": "sealake",
    "lakefs_ref": "main",  # 可以是 branch、tag 或 commit ID（推荐用 commit ID 锁定版本）

    # remote→local 映射
    # 5090 上的本地数据存储路径
    "local_base_dir": r"C:\Users\admin\Desktop\Unet\dataset",

    # LakeFS 上的子路径 → 本地子目录
    "data_mapping": {
         "raw": "raw",
    },

    # 是否强制重新下载（即使本地已存在）
    "force_redownload": False,
}
config = task.connect(config)

# ===== 发到远程 Agent（5090）执行 =====
task.execute_remotely(queue_name="default", exit_process=True)

# ===== 下面的代码在 5090 上跑 =====
import boto3
from botocore.config import Config
from pathlib import Path
from tqdm import tqdm


def get_s3_client(endpoint):
    """从 Agent 环境读取凭据，创建 LakeFS S3 兼容客户端。"""
    access_key = os.environ.get("LAKEFS_ACCESS_KEY_ID")
    secret_key = os.environ.get("LAKEFS_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        raise RuntimeError(
            "Set LAKEFS_ACCESS_KEY_ID and LAKEFS_SECRET_ACCESS_KEY "
            "in the ClearML Agent environment before downloading data."
        )
    return boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(
            signature_version='s3v4',
            retries={'max_attempts': 3},
        ),
        region_name='us-east-1',  # LakeFS 要求设一个，但不重要
    )


def download_prefix(s3_client, repo, ref, remote_prefix, local_dir, force=False):
    """从 LakeFS 下载某个 prefix 下的所有文件"""
    # LakeFS S3 路径格式: s3://<repo>/<ref>/<path>
    # 用 boto3 list_objects_v2 + get_object
    bucket = repo
    key_prefix = f"{ref}/{remote_prefix.strip('/')}/"

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Downloading {remote_prefix} → {local_dir} ===")

    # 列出所有对象
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=key_prefix)

    all_keys = []
    for page in pages:
        for obj in page.get('Contents', []):
            all_keys.append(obj['Key'])

    if not all_keys:
        print(f"[WARN] No files found under s3://{bucket}/{key_prefix}")
        return 0

    print(f"Found {len(all_keys)} files")

    downloaded = 0
    skipped = 0
    for key in tqdm(all_keys, desc=remote_prefix):
        # 去掉 ref/prefix 部分，得到文件相对名
        rel_path = key[len(key_prefix):]
        if not rel_path:  # 跳过目录本身
            continue

        local_path = local_dir / rel_path

        # 已存在且不强制 → 跳过
        if local_path.exists() and not force:
            skipped += 1
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(bucket, key, str(local_path))
        downloaded += 1

    print(f"[OK] {remote_prefix}: downloaded {downloaded}, skipped {skipped}")
    return downloaded


def main():
    print("=" * 60)
    print("LakeFS Data Download Task")
    print("=" * 60)
    print(f"Endpoint: {config['lakefs_endpoint']}")
    print(f"Repo:     {config['lakefs_repo']}")
    print(f"Ref:      {config['lakefs_ref']}")
    print(f"Local:    {config['local_base_dir']}")
    print()

    s3 = get_s3_client(config['lakefs_endpoint'])

    # 验证连接
    try:
        s3.head_bucket(Bucket=config['lakefs_repo'])
        print(f"[OK] Connected to LakeFS repo: {config['lakefs_repo']}")
    except Exception as e:
        print(f"[FAIL] Cannot access repo: {e}")
        raise

    total = 0
    for remote_path, local_subdir in config['data_mapping'].items():
        local_dir = os.path.join(config['local_base_dir'], local_subdir)
        n = download_prefix(
            s3,
            repo=config['lakefs_repo'],
            ref=config['lakefs_ref'],
            remote_prefix=remote_path,
            local_dir=local_dir,
            force=config['force_redownload'],
        )
        total += n

    print(f"\n{'=' * 60}")
    print(f"Total downloaded: {total} files")
    print(f"Data ready at: {config['local_base_dir']}")
    print(f"{'=' * 60}")

    # ===== 关键：把本地数据路径记录到 task，方便训练任务读取 =====
    task.set_user_properties(
        local_data_root=config['local_base_dir'],
        lakefs_ref=config['lakefs_ref'],
        files_downloaded=total,
    )

    # 同时记录为 artifact（更结构化的方式）
    task.upload_artifact(
        name='data_info',
        artifact_object={
            'local_path': config['local_base_dir'],
            'lakefs_repo': config['lakefs_repo'],
            'lakefs_ref': config['lakefs_ref'],
            'data_mapping': config['data_mapping'],
        },
    )

    print("[OK] Task metadata recorded")


if __name__ == "__main__":
    main()
