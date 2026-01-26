import pandas as pd
import numpy as np
from pathlib import Path

def analyze_parquet_file(file_path):
    """
    分析Parquet文件的主要信息
    
    Args:
        file_path (str): Parquet文件路径
    """
    print("=" * 60)
    print(f"Parquet文件分析: {file_path}")
    print("=" * 60)
    
    try:
        # 读取Parquet文件
        print("正在读取文件...")
        df = pd.read_parquet(file_path)
        
        # 基本信息
        print("\n📊 基本信息:")
        print(f"文件大小: {Path(file_path).stat().st_size / (1024*1024):.2f} MB")
        print(f"行数: {len(df):,}")
        print(f"列数: {len(df.columns)}")
        print(f"数据类型: {type(df)}")
        
        # 列信息
        print("\n📋 列信息:")
        print("-" * 40)
        for i, col in enumerate(df.columns, 1):
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            unique_count = df[col].nunique()
            
            print(f"{i:2d}. {col:<20} | 类型: {dtype:<10} | "
                  f"缺失: {null_count:,} ({null_pct:.1f}%) | "
                  f"唯一值: {unique_count:,}")
        
        # 内存使用情况
        print("\n💾 内存使用情况:")
        memory_usage = df.memory_usage(deep=True).sum() / (1024*1024)
        print(f"总内存使用: {memory_usage:.2f} MB")
        
        # 显示每列的内存使用
        print("\n每列内存使用 (MB):")
        col_memory = df.memory_usage(deep=True) / (1024*1024)
        for col, mem in col_memory.items():
            print(f"  {col:<20}: {mem:.2f}")
        
        # 数据样本
        print("\n🔍 数据样本 (前5行):")
        print("-" * 40)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        print(df.head())
        
        # 数据类型分布
        print("\n🏷️ 数据类型分布:")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} 列")
        
        # 数值列的统计信息
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print("\n📈 数值列统计信息:")
            print("-" * 40)
            print(df[numeric_cols].describe())
        
        # 文本列的统计信息
        text_cols = df.select_dtypes(include=['object', 'string']).columns
        if len(text_cols) > 0:
            print("\n📝 文本列统计信息:")
            print("-" * 40)
            for col in text_cols:
                if df[col].notna().any():
                    avg_length = df[col].dropna().str.len().mean()
                    max_length = df[col].dropna().str.len().max()
                    print(f"{col}:")
                    print(f"      平均长度: {avg_length:.1f}")
                    print(f"      最大长度: {max_length}")
                    print()
        
        # 警告信息
        warnings = []
        if df.isnull().any().any():
            null_cols = df.columns[df.isnull().any()].tolist()
            warnings.append(f"⚠️ 发现缺失值列: {null_cols}")
        
        high_null_cols = df.columns[(df.isnull().sum() / len(df)) > 0.5].tolist()
        if high_null_cols:
            warnings.append(f"⚠️ 缺失值比例>50%的列: {high_null_cols}")
        
        if warnings:
            print("\n⚠️ 警告信息:")
            for warning in warnings:
                print(f"  {warning}")
        
        print("\n✅ 分析完成!")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {str(e)}")
        print("请检查文件路径是否正确，以及是否已安装pandas和pyarrow库")

def main():
    # 使用示例
    file_path = "training_data/SFT_data/sft_data.parquet"  # 请根据实际路径修改
    
    # 检查文件是否存在
    if not Path(file_path).exists():
        print(f"❌ 文件不存在: {file_path}")
        print("请检查文件路径是否正确")
        return
    
    analyze_parquet_file(file_path)

if __name__ == "__main__":
    main()