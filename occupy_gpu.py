# save as occupy_gpu.py
import argparse
import math
import time
import torch


def occupy_memory(gpu_ids, per_gpu_gb, chunk_gb=0.25, sleep_sec=1.0, hold=True):
    per_gpu_mb = per_gpu_gb * 1024
    chunk_mb = chunk_gb * 1024
    target_bytes = per_gpu_mb * 1024 ** 2
    chunk_bytes = chunk_mb * 1024 ** 2
    holders = {gid: [] for gid in gpu_ids}

    print(
        f"目标: 每张卡约占用 {per_gpu_gb} GB (分块 {chunk_gb} GB ≈ {chunk_mb:.0f} MB)，按 Ctrl+C 退出。"
    )

    for gid in gpu_ids:
        torch.cuda.set_device(gid)
        print(f"\n开始占用 cuda:{gid} ...")
        while torch.cuda.memory_allocated() < target_bytes:
            allocated = torch.cuda.memory_allocated()
            remaining = max(0, target_bytes - allocated)
            this_chunk = min(chunk_bytes, remaining)
            if this_chunk < 4:  # 至少分配 1 个 float32
                break
            num_elems = math.ceil(this_chunk / 4)
            try:
                holders[gid].append(
                    torch.empty(
                        num_elems, dtype=torch.float32, device=f"cuda:{gid}"
                    )
                )
            except RuntimeError as e:
                print(f"  分配失败（可能 OOM）: {e}")
                break
            now_gb = torch.cuda.memory_allocated() / 1024 ** 3
            print(
                f"  当前已占用: {now_gb:.2f} GB / 目标 {per_gpu_gb} GB", end="\r"
            )
        used_gb = torch.cuda.memory_allocated() / 1024 ** 3
        print(f"\n  cuda:{gid} 完成，占用约 {used_gb:.2f} GB")

    if hold:
        try:
            while True:
                time.sleep(sleep_sec)
        except KeyboardInterrupt:
            print("\n收到中断，释放显存后退出。")
    holders.clear()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="占用指定 GPU 显存的小工具（按 GB）")
    parser.add_argument(
        "--gpus", type=int, nargs="+", required=True, help="要占用的 GPU 编号，如: --gpus 0 1"
    )
    parser.add_argument(
        "--per-gpu-gb", type=float, required=True, help="每张卡要占用的显存 GB 数"
    )
    parser.add_argument(
        "--chunk-gb", type=float, default=0.25, help="分配粒度（GB），默认 0.25 即约 256 MB"
    )
    parser.add_argument("--sleep", type=float, default=1.0, help="保持时的休眠秒数")
    parser.add_argument(
        "--no-hold",
        action="store_true",
        help="分配完立即退出（默认会一直占用，需 Ctrl+C 释放）",
    )
    args = parser.parse_args()
    occupy_memory(
        args.gpus,
        args.per_gpu_gb,
        args.chunk_gb,
        args.sleep,
        hold=not args.no_hold,
    )


if __name__ == "__main__":
    main()