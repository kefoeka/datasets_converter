import pyarrow.parquet as pq
import pandas as pd

# 读取 parquet 文件
try:
    # 读取单个 parquet 文件
    df = pq.read_table('train-00000-of-00001.parquet').to_pandas()
    # df = pq.read_table('train-00000-of-00002.parquet').to_pandas()
    
    # 显示基本信息
    print("数据集形状:", df.shape)
    print("数据类型:\n", df.dtypes)
    
    # 查看 parquet 所有列名及前5行数据
    print("\n列名:", df.columns.tolist())
    print("\n前5行数据预览:")
    print(df.head())
    
    # 查看某一列前5行数据
    print("\n'deepseek_grade_reason'列前5行数据:")
    print(df['deepseek_grade_reason'].head())
    
    # 基本统计信息
    print("\n数值列统计信息:")
    print(df.describe())
    
except FileNotFoundError:
    print("文件不存在，请检查文件路径")
except Exception as e:
    print(f"读取文件时出错: {e}")

