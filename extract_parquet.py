from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Callable, List

import argparse


def parquet_head_to_new(src_path: str, dst_path: str, n: int) -> None:
    """
    从 src_path 抽取前 n 行写入 dst_path。
    """
    if n <= 0:
        raise ValueError("n 必须为正整数")
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"找不到文件: {src}")

    reader = pq.ParquetFile(src)
    needed = n
    tables = []

    for i in range(reader.num_row_groups):
        rg = reader.read_row_group(i)
        if needed < len(rg):
            rg = rg.slice(0, needed)
        tables.append(rg)
        needed -= len(rg)
        if needed <= 0:
            break

    if not tables:
        raise ValueError("源文件为空，无法抽取")
    result = pa.concat_tables(tables)
    pq.write_table(result, dst_path)
    print(f"已写出前 {n} 行到: {dst_path}")


def filter_by_data_source_suffix(data_source: str, allowed_suffixes: List[str]) -> bool:
    """
    默认的抽取逻辑：检查data_source的后缀是否在允许的后缀列表中。

    Args:
        data_source: 数据源字符串
        allowed_suffixes: 允许的后缀列表

    Returns:
        bool: 是否满足条件
    """
    if not isinstance(data_source, str):
        return False
    return any(data_source.endswith(suffix) for suffix in allowed_suffixes)


def extract_parquet_by_condition(
    src_path: str,
    dst_path: str,
    filter_func: Callable[[str], bool],
    data_source_column: str = "data_source"
) -> None:
    """
    从parquet文件中根据条件抽取数据并写入新文件。

    Args:
        src_path: 源parquet文件路径
        dst_path: 目标parquet文件路径
        filter_func: 过滤函数，接受data_source值，返回bool
        data_source_column: data_source列名，默认为"data_source"
    """
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"找不到文件: {src}")

    # 读取整个文件
    table = pq.read_table(src)

    # 检查是否存在指定的列
    if data_source_column not in table.column_names:
        raise ValueError(f"文件中不存在列 '{data_source_column}'，可用列: {table.column_names}")

    # 获取data_source列的值
    data_source_values = table.column(data_source_column).to_pylist()

    # 应用过滤条件
    mask = [filter_func(value) for value in data_source_values]

    # 过滤数据
    filtered_table = table.filter(mask)

    if len(filtered_table) == 0:
        print(f"警告：没有找到满足条件的数据")
        return

    # 写入新文件
    pq.write_table(filtered_table, dst_path)
    print(f"已抽取 {len(filtered_table)} 行数据（总共 {len(table)} 行）到: {dst_path}")

def main():
    parser = argparse.ArgumentParser(description="占用指定 GPU 显存的小工具（按 GB）")
    parser.add_argument(
        "--src_path", type=str, required=True, help="源parquet文件路径"
    )
    parser.add_argument(
        "--dst_path", type=str, required=True, help="目标parquet文件路径"
    )
    args = parser.parse_args()
    src_path = args.src_path
    dst_path = args.dst_path
    
    def easy_hard_filter(data_source: str) -> bool:
        return filter_by_data_source_suffix(data_source, ["easy", "hard"])

    extract_parquet_by_condition(
        src_path=src_path,
        dst_path=dst_path,
        filter_func=easy_hard_filter,
    )
if __name__ == "__main__":
    main()
    