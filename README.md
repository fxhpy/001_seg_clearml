# 001_seg_clearml

使用 ClearML 调度 LakeFS 数据下载和图像语义分割训练。训练模型为 DeepLabV3Plus，默认使用 MobileNet V2 编码器、ImageNet 预训练权重、4 个类别，输入缩放为 512 × 512。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `download_data.py` | 从 LakeFS 的 S3 兼容接口下载数据，向 ClearML 记录数据位置和版本 |
| `dataset.py` | 读取图像、掩码及进行数据增强 |
| `train.py` | 通过 ClearML 查找下载任务并执行训练 |
| `requirements.txt` | Python 依赖 |
| `.env.example` | 环境变量名称示例，不包含真实凭据 |

## 配置与运行

在 PowerShell 中创建环境并安装依赖：

```powershell
Set-Location 'D:\Desktop\001_seg_clearml'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\clearml-init.exe
```

ClearML Agent 需要单独配置，并监听 `default` 队列。两个脚本都会提交远程任务；实际数据下载和训练发生在 Agent 上。

在启动 **ClearML Agent 的机器和进程环境**中设置以下变量，值使用自己的 LakeFS 凭据：

```powershell
$env:LAKEFS_ACCESS_KEY_ID = '<your-access-key>'
$env:LAKEFS_SECRET_ACCESS_KEY = '<your-secret-key>'
```

`.env.example` 只是配置参考，脚本不会自动读取 `.env` 文件。远程 Agent 也不会自动继承提交任务的电脑上的环境变量。真实密钥应在 Agent 上配置，不能写入源码或 ClearML 任务参数。

按实际环境调整 `download_data.py` 中的 `lakefs_endpoint`、`lakefs_repo`、`lakefs_ref`、`local_base_dir` 和 `data_mapping`。训练代码需要以下数据结构：

```text
<local_base_dir>/
  Train/
    imgs/
    masks/
  Val/
    imgs/
    masks/
```

当前下载映射是 `raw` → `local_base_dir/raw`，而训练直接读取 `local_base_dir/Train` 和 `local_base_dir/Val`。运行前必须根据实际 LakeFS 目录调整映射或数据根目录。图像与掩码也需要一一对应。

先提交下载任务，确认其在 ClearML 中成功完成，再提交训练任务：

```powershell
.\.venv\Scripts\python.exe download_data.py
# 等待下载任务完成后执行：
.\.venv\Scripts\python.exe train.py
```

`train.py` 中的 `data_task_id` 留空时查找已完成的下载任务，也可以指定任务 ID。训练权重写入 Agent 上的 `checkpoints/`，TensorBoard 日志写入 `runs/`。

## 首次上传 GitHub 的流程

Git 负责本地版本记录，GitHub 保存远程仓库。流程是：检查文件 → 初始化或恢复本地仓库 → 本地提交 → 登录 GitHub → 创建或连接远程仓库 → 推送 → 核对结果。

### 1. 检查文件与忽略规则

只提交源代码、依赖和说明。数据集、模型权重、虚拟环境、日志、`.env`、`clearml.conf` 和 `.git.bak/` 均由 `.gitignore` 排除。提交前检查 `git status` 和暂存区差异，确保没有真实密钥。

### 2. 准备本地仓库

本项目已有 Git 历史，直接使用现有仓库，不必重新初始化。对完全没有 Git 历史的新项目，才执行 `git init -b main`。

在项目目录执行（姓名和邮箱只在没有正确配置时设置，使用自己的信息）：

```powershell
git status
git config user.name
git config user.email
# 仅在未配置或配置错误时执行，并替换为自己的信息：
# git config user.name 'YOUR_NAME'
# git config user.email 'YOUR_EMAIL'
git add .gitignore .env.example README.md dataset.py download_data.py requirements.txt train.py
git diff --cached --stat
git diff --cached
git commit -m "Prepare project for GitHub"
```

### 3. 登录 GitHub

本次操作使用官方便携版 GitHub CLI。若 `gh` 不在 PATH 中，先在当前终端运行：

```powershell
Set-Alias gh "$env:USERPROFILE\.codex\tools\github-cli\2.102.1\bin\gh.exe"
```

在其他电脑上，可用 `winget install --id GitHub.cli --exact --source winget` 安装 GitHub CLI，然后重新打开 PowerShell，无需设置上述别名。

接着登录，浏览器中的 GitHub 登录和授权由账号本人完成：

```powershell
gh auth login --hostname github.com --git-protocol https --web
gh auth status
gh auth setup-git --hostname github.com
```

### 4. 创建或连接远程仓库

如果还没有远程仓库，下面是**新建私有仓库**的示例。先将 `YOUR_GITHUB_USERNAME` 替换为登录的用户名；确定公开时将 `--private` 改为 `--public`：

```powershell
gh repo create YOUR_GITHUB_USERNAME/001_seg_clearml --private --source . --remote origin --push
```

如果已在 GitHub 创建**空仓库**，使用下面这条路径（与上一条二选一）：

```powershell
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/001_seg_clearml.git
git push -u origin HEAD
```

已有 `origin` 时先用 `git remote -v` 核对地址。已有内容的远程仓库需要先检查并整合历史；不要用强制推送覆盖。

### 5. 验证上传

```powershell
git status --short --branch
git remote -v
git rev-parse HEAD
$currentBranch = git branch --show-current
git ls-remote origin "refs/heads/$currentBranch"
```

本地工作区应干净，当前分支应跟踪远程分支，最后两条命令输出的提交哈希应一致。最后打开 GitHub 仓库，核对源码和 README 已显示。

## 后续更新

每次修改后，在项目目录执行：

```powershell
git status
git add <本次需要上传的文件路径>
git diff --cached
git commit -m "描述这次修改"
git push
```

尖括号中的内容需要替换为真实路径。上传其他项目时重复上述流程，每个项目连接自己的 GitHub 仓库。

参考：[GitHub 官方本地项目上传说明](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)、[GitHub CLI 登录](https://cli.github.com/manual/gh_auth_login)、[创建仓库](https://cli.github.com/manual/gh_repo_create)。
