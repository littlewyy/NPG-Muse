from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
import glob
from typing import List, Union


def merge_parquet_files(
    input_paths: List[str],
    output_path: str,
    ignore_schema_mismatch: bool = False
) -> None:
    """
    合并多个parquet文件为一个文件。

    Args:
        input_paths: 输入文件路径列表，支持通配符
        output_path: 输出文件路径
        ignore_schema_mismatch: 是否忽略schema不匹配的错误
    """
    # 展开通配符并收集所有文件路径
    all_files = []
    for path_pattern in input_paths:
        matched_files = glob.glob(path_pattern)
        if not matched_files:
            print(f"警告: 找不到匹配的文件: {path_pattern}")
            continue
        all_files.extend(matched_files)

    if not all_files:
        raise ValueError("没有找到任何有效的输入文件")

    print(f"找到 {len(all_files)} 个文件待合并:")
    for f in all_files:
        print(f"  {f}")

    # 读取第一个文件以确定schema
    first_file = all_files[0]
    print(f"\n读取第一个文件: {first_file}")
    first_table = pq.read_table(first_file)
    base_schema = first_table.schema
    print(f"基准schema包含 {len(base_schema)} 列: {[field.name for field in base_schema]}")

    # 收集所有表的列表
    tables = [first_table]
    total_rows = len(first_table)

    # 读取其余文件
    for file_path in all_files[1:]:
        print(f"读取文件: {file_path}")
        try:
            table = pq.read_table(file_path)
            current_schema = table.schema

            # 检查schema兼容性
            if not ignore_schema_mismatch:
                if not base_schema.equals(current_schema):
                    print(f"警告: 文件 {file_path} 的schema与基准schema不匹配")
                    print(f"基准schema: {[f'{field.name}:{field.type}' for field in base_schema]}")
                    print(f"当前schema: {[f'{field.name}:{field.type}' for field in current_schema]}")

                    # 尝试统一schema
                    try:
                        table = pa.Table.from_arrays(
                            [table.column(field.name) if field.name in table.column_names
                             else pa.array([None] * len(table), type=field.type)
                             for field in base_schema],
                            names=[field.name for field in base_schema]
                        )
                        print(f"已统一 {file_path} 的schema")
                    except Exception as e:
                        raise ValueError(f"无法统一schema: {e}")

            tables.append(table)
            total_rows += len(table)
            print(f"  添加了 {len(table)} 行")

        except Exception as e:
            print(f"错误: 读取文件 {file_path} 失败: {e}")
            raise

    print(f"\n总计读取了 {total_rows} 行数据")

    # 合并所有表
    print("开始合并表...")
    merged_table = pa.concat_tables(tables, promote_options="default")

    # 验证合并结果
    if len(merged_table) != total_rows:
        raise ValueError(f"合并后行数不匹配: 期望 {total_rows}, 实际 {len(merged_table)}")

    # 写入合并后的文件
    print(f"写入合并后的文件: {output_path}")
    pq.write_table(merged_table, output_path)

    print("合并完成!")
    print(f"输出文件: {output_path}")
    print(f"总行数: {len(merged_table)}")
    print(f"总列数: {len(merged_table.schema)}")


def main():
    parser = argparse.ArgumentParser(
        description="合并多个Parquet文件为一个文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 合并指定文件
  python merge_parquet.py --input file1.parquet file2.parquet --output merged.parquet

  # 使用通配符合并文件
  python merge_parquet.py --input "data/*.parquet" --output merged.parquet

  # 忽略schema不匹配
  python merge_parquet.py --input *.parquet --output merged.parquet --ignore-schema-mismatch
        """
    )

    parser.add_argument(
        "--input", "-i", nargs="+", required=True,
        help="输入文件路径，支持通配符（如 *.parquet 或 data/*.parquet）"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="输出文件路径"
    )
    parser.add_argument(
        "--ignore-schema-mismatch", action="store_true",
        help="忽略schema不匹配的错误，尝试自动统一schema"
    )

    args = parser.parse_args()

    try:
        merge_parquet_files(
            input_paths=args.input,
            output_path=args.output,
            ignore_schema_mismatch=args.ignore_schema_mismatch
        )
    except Exception as e:
        print(f"错误: {e}")
        exit(1)


if __name__ == "__main__":
    main()