"""
一个支持断点续传与镜像源配置的 HuggingFace 下载脚本。
依赖: huggingface_hub (pip install huggingface_hub)
"""

import argparse
import os
import sys
from typing import List, Optional

from huggingface_hub import snapshot_download


def download_repo(
    repo_id: str,
    repo_type: str,
    local_dir: str,
    *,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    mirror_endpoint: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
    max_workers: int = 8,
) -> str:
    """
    下载 HuggingFace 模型或数据集，支持断点续传与可选镜像源。
    返回实际写入内容的本地目录。
    """
    # 处理镜像: snapshot_download 默认读取环境变量 HF_ENDPOINT
    original_endpoint = os.environ.get("HF_ENDPOINT")
    if mirror_endpoint:
        os.environ["HF_ENDPOINT"] = mirror_endpoint.rstrip("/")

    try:
        target_dir = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            resume_download=True,  # 启用断点续传
            max_workers=max_workers,
        )
        return target_dir
    finally:
        # 恢复环境变量，避免影响后续进程
        if mirror_endpoint is not None:
            if original_endpoint is None:
                os.environ.pop("HF_ENDPOINT", None)
            else:
                os.environ["HF_ENDPOINT"] = original_endpoint

# python hf_downloader.py Qwen/Qwen2.5-7B-Instruct --repo-type model --mirror-endpoint https://hf-mirror.com --local-dir /hpc2hdd/home/mpeng885/models/Qwen
# python hf_downloader.py Qwen/Qwen3-8B-Base --repo-type model --mirror-endpoint https://hf-mirror.com --local-dir /hpc2hdd/home/mpeng885/models/Qwen
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="下载 HuggingFace 模型或数据集，支持断点续传与镜像源"
    )
    parser.add_argument("repo_id", help="仓库 ID，如 'bert-base-uncased' 或 'glue'")
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=["model", "dataset"],
        help="仓库类型，默认为 model",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="分支/Tag/Commit SHA，可选，默认使用仓库默认分支",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="访问私有仓库时的 token，未提供则尝试读取环境变量 HF_TOKEN",
    )
    parser.add_argument(
        "--mirror-endpoint",
        default=os.environ.get("HF_ENDPOINT"),
        help="镜像源根地址，如 https://hf-mirror.com；默认沿用环境变量 HF_ENDPOINT",
    )
    parser.add_argument(
        "--allow",
        nargs="*",
        default=None,
        help="仅下载匹配的文件模式列表（glob），例如 *.bin tokenizer.json",
    )
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=None,
        help="忽略匹配的文件模式列表（glob）",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help="写入目录；默认位于 ./downloads/<repo_id>/",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="自定义缓存目录；默认使用 huggingface_hub 的默认缓存位置",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="下载并发线程数，默认为 8",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # 默认 local_dir
    local_dir = args.local_dir
    if not local_dir:
        sanitized = args.repo_id.replace("/", "_")
        local_dir = os.path.join("downloads", sanitized)
    os.makedirs(local_dir, exist_ok=True)

    try:
        target = download_repo(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            token=args.token,
            mirror_endpoint=args.mirror_endpoint,
            allow_patterns=args.allow,
            ignore_patterns=args.ignore,
            local_dir=local_dir,
            cache_dir=args.cache_dir,
            max_workers=args.max_workers,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"下载失败: {exc}", file=sys.stderr)
        return 1

    print(f"下载完成，已写入: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
