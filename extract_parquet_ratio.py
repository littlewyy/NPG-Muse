from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
from typing import Optional


def extract_parquet_by_ratio(
    src_path: str,
    dst_path: str,
    ratio: float,
    data_source_column: str = "data_source",
    random_state: Optional[int] = None
) -> None:
    """
    从parquet文件中按照data_source分组，然后对每个组随机提取指定比例的数据。

    Args:
        src_path: 源parquet文件路径
        dst_path: 目标parquet文件路径
        ratio: 采样比例，0.0-1.0之间的小数
        data_source_column: data_source列名，默认为"data_source"
        random_state: 随机种子，用于重现结果，默认为None
    """
    if not (0 < ratio <= 1):
        raise ValueError("ratio 必须在 0 到 1 之间")

    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"找不到文件: {src}")

    print(f"开始读取文件: {src_path}")

    # 使用pandas读取parquet文件，方便分组和采样
    df = pd.read_parquet(src_path)

    # 检查是否存在指定的列
    if data_source_column not in df.columns:
        raise ValueError(f"文件中不存在列 '{data_source_column}'，可用列: {list(df.columns)}")

    print(f"原始数据行数: {len(df)}")
    print(f"按 {data_source_column} 分组统计:")

    # 统计每个组的大小
    group_counts = df[data_source_column].value_counts()
    for data_source, count in group_counts.items():
        print(f"  {data_source}: {count} 行")

    # 对每个组进行采样
    sampled_dfs = []
    total_sampled = 0

    for data_source, group_df in df.groupby(data_source_column):
        group_size = len(group_df)
        sample_size = int(group_size * ratio)

        # 确保至少采样1行（如果原组不为空）
        if group_size > 0 and sample_size == 0:
            sample_size = 1

        print(f"对组 '{data_source}' 采样: {sample_size}/{group_size} 行")

        # 随机采样
        sampled_group = group_df.sample(
            n=sample_size,
            random_state=random_state,
            replace=False
        )
        sampled_dfs.append(sampled_group)
        total_sampled += sample_size

    # 合并所有采样的数据
    if sampled_dfs:
        result_df = pd.concat(sampled_dfs, ignore_index=True)

        # 写入新文件
        result_df.to_parquet(dst_path, index=False)

        print(f"\n采样完成!")
        print(f"输出文件: {dst_path}")
        print(f"采样后数据行数: {len(result_df)} (原始: {len(df)})")

        # 显示采样后的分组统计
        print("采样后按组统计:")
        sampled_counts = result_df[data_source_column].value_counts()
        for data_source, count in sampled_counts.items():
            original_count = group_counts.get(data_source, 0)
            print(f"  {data_source}: {count} 行 (原始: {original_count} 行)")
    else:
        print("警告：没有采样到任何数据")


def main():
    parser = argparse.ArgumentParser(
        description="从parquet文件中按data_source分组随机采样指定比例的数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python extract_parquet_ratio.py --src_path input.parquet --dst_path output.parquet --ratio 0.6666666666666667
  python extract_parquet_ratio.py --src_path input.parquet --dst_path output.parquet --ratio 0.5 --random_state 42
        """
    )

    parser.add_argument(
        "--src_path", type=str, required=True,
        help="源parquet文件路径"
    )
    parser.add_argument(
        "--dst_path", type=str, required=True,
        help="目标parquet文件路径"
    )
    parser.add_argument(
        "--ratio", type=float, required=True,
        help="采样比例，0.0-1.0之间的浮点数，例如 0.67 表示采样 2/3 的数据"
    )
    parser.add_argument(
        "--data_source_column", type=str, default="data_source",
        help="data_source列名，默认为 'data_source'"
    )
    parser.add_argument(
        "--random_state", type=int, default=42,
        help="随机种子，用于重现采样结果，默认为None（每次运行结果不同）"
    )

    args = parser.parse_args()

    try:
        extract_parquet_by_ratio(
            src_path=args.src_path,
            dst_path=args.dst_path,
            ratio=args.ratio,
            data_source_column=args.data_source_column,
            random_state=args.random_state
        )
    except Exception as e:
        print(f"错误: {e}")
        exit(1)


if __name__ == "__main__":
    main()