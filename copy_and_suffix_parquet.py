from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.parquet as pq


def _append_suffix_to_column(table: pa.Table, column_name: str, suffix: str) -> pa.Table:
    if column_name not in table.column_names:
        raise ValueError(f"文件中不存在列 '{column_name}'，可用列: {table.column_names}")

    col_index = table.column_names.index(column_name)
    col = table.column(column_name)

    values = col.to_pylist()
    updated_values = [
        None if value is None else f"{value}{suffix}" for value in values
    ]

    if pa.types.is_string(col.type) or pa.types.is_large_string(col.type):
        updated_col = pa.array(updated_values, type=col.type)
    else:
        updated_col = pa.array(updated_values, type=pa.string())

    return table.set_column(col_index, column_name, updated_col)


def copy_and_suffix_parquet_files(
    src_dir: str, dst_dir: str, suffix: str, column_name: str = "data_source"
) -> None:
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    if not src_path.exists():
        raise FileNotFoundError(f"找不到源文件夹: {src_path}")
    if not src_path.is_dir():
        raise NotADirectoryError(f"源路径不是文件夹: {src_path}")

    dst_path.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(src_path.glob("*.parquet"))
    if not parquet_files:
        print(f"源文件夹中未找到 parquet 文件: {src_path}")
        return

    for parquet_file in parquet_files:
        target_file = dst_path / parquet_file.name
        shutil.copy2(parquet_file, target_file)

        table = pq.read_table(target_file)
        table = _append_suffix_to_column(table, column_name, suffix)
        pq.write_table(table, target_file)

        print(f"已处理: {target_file}")

    print(f"完成：共处理 {len(parquet_files)} 个 parquet 文件")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="复制parquet文件并给data_source追加后缀"
    )
    parser.add_argument(
        "--src_dir", type=str, required=True, help="源parquet文件夹路径"
    )
    parser.add_argument(
        "--dst_dir", type=str, required=True, help="目标parquet文件夹路径"
    )
    parser.add_argument(
        "--suffix", type=str, required=True, help="要追加到data_source的后缀"
    )
    parser.add_argument(
        "--column",
        type=str,
        default="data_source",
        help="需要追加后缀的列名，默认 data_source",
    )
    args = parser.parse_args()

    copy_and_suffix_parquet_files(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        suffix=args.suffix,
        column_name=args.column,
    )


if __name__ == "__main__":
    main()
