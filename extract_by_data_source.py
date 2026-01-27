import argparse
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
from typing import List

def extract_by_data_source(
    src_path: str,
    dst_path: str,
    data_sources: List[str],
    data_source_column: str = "data_source"
) -> None:
    """
    从parquet文件中提取data_source列值为指定列表中的行。

    Args:
        src_path: 源parquet文件路径
        dst_path: 目标parquet文件路径
        data_sources: 需要提取的data_source值列表
        data_source_column: data_source所在的列名
    """
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"找不到源文件: {src}")

    print(f"正在读取文件: {src_path}")
    # 读取整个文件
    table = pq.read_table(src)

    # 检查列是否存在
    if data_source_column not in table.column_names:
        raise ValueError(f"列 '{data_source_column}' 不存在。可用列: {table.column_names}")

    # 获取该列的值并进行过滤
    # 使用 pyarrow 的 compute 模块进行高效过滤
    import pyarrow.compute as pc
    
    # 构建过滤表达式: data_source_column is in data_sources
    mask = pc.is_in(table.column(data_source_column), value_set=pa.array(data_sources))
    
    filtered_table = table.filter(mask)

    if len(filtered_table) == 0:
        print(f"警告：未找到 data_source 为 {data_sources} 的数据。")
        return

    # 写入新文件
    pq.write_table(filtered_table, dst_path)
    print(f"成功抽取 {len(filtered_table)} 行数据到: {dst_path}")
    print(f"原始行数: {len(table)}, 抽取比例: {len(filtered_table)/len(table):.2%}")

def main():
    parser = argparse.ArgumentParser(description="按 data_source 的特定值提取 Parquet 数据")
    parser.add_argument("--src", type=str, required=True, help="输入 Parquet 文件路径")
    parser.add_argument("--dst", type=str, required=True, help="输出 Parquet 文件路径")
    parser.add_argument("--sources", type=str, nargs="+", required=True, help="需要提取的 data_source 值列表（空格分隔）")
    parser.add_argument("--column", type=str, default="data_source", help="data_source 所在的列名，默认为 'data_source'")

    args = parser.parse_args()

    try:
        extract_by_data_source(
            src_path=args.src,
            dst_path=args.dst,
            data_sources=args.sources,
            data_source_column=args.column
        )
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()
