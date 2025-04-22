#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据集加载示例
从Hugging Face加载数据集并保存到本地
"""

import os
import argparse
import traceback
import json
from datasets import load_dataset

# 定义本地数据集保存路径
LOCAL_DATASET_PATH = "local_datasets"

def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='从Hugging Face加载数据集并保存到本地')
    parser.add_argument('-d', '--dataset_name', type=str, required=True,
                       help='数据集名称（必需），例如：togethercomputer/RedPajama-Data-1T-Sample')
    parser.add_argument('-o', '--output_dir', type=str, default=LOCAL_DATASET_PATH,
                       help='输出目录路径，默认为 local_datasets')
    return parser.parse_args()

def log_error_with_traceback(error_msg: str, e: Exception) -> None:
    """记录错误信息和完整的堆栈跟踪"""
    print(f"\n错误: {error_msg}")
    print(f"错误类型: {type(e).__name__}")
    print(f"错误信息: {str(e)}")
    print("错误堆栈:")
    print(traceback.format_exc())

def show_first_n_lines(file_path: str, n: int = 5) -> None:
    """显示文件的前n行内容"""
    print(f"\n显示保存文件的前{n}条数据:")
    print("="*80)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                data = json.loads(line)
                print(f"\n第{i+1}条数据:")
                for key, value in data.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"{key}: {value[:100]}...")
                    else:
                        print(f"{key}: {value}")
                print("-"*40)
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
    print("="*80)

def main() -> None:
    """主函数"""
    args = parse_arguments()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # 加载数据集
        print(f"正在加载数据集: {args.dataset_name}")
        ds = load_dataset(args.dataset_name)
        
        # 打印数据集信息
        print("\n数据集信息:")
        print("数据集分割:", ds.keys())
        print("数据集类型:", type(ds))
        
        for split in ds.keys():
            print(f"\n{split} 分割信息:")
            print(f"特征:", ds[split].features)
            print(f"大小:", len(ds[split]))
            print(f"示例:", ds[split][0] if len(ds[split]) > 0 else "空")
        
        # 保存训练集
        output_file = os.path.join(args.output_dir, f"{args.dataset_name.split('/')[-1]}.json")
        print(f"\n正在保存数据集到: {output_file}")
        ds['train'].to_json(output_file, lines=True)
        print("保存完成！")
        
        # 显示保存文件的前5条数据
        show_first_n_lines(output_file, 5)
        
    except Exception as e:
        log_error_with_traceback("加载或保存数据集时发生错误", e)
        raise

if __name__ == "__main__":
    main()
