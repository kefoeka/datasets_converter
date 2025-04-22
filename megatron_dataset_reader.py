#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Megatron数据集读取工具
用于读取和验证Megatron格式的数据集
"""

# 标准库导入
import os
import sys
import argparse
import traceback
from pathlib import Path

# 第三方库导入
from transformers import AutoTokenizer

# 本地导入
MEGATRON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Megatron-LM")
sys.path.append(MEGATRON_PATH)
from megatron.core.datasets import indexed_dataset

def log_error_with_traceback(error_msg: str, e: Exception) -> None:
    """记录错误信息和完整的堆栈跟踪"""
    print("\n" + "="*80)
    print(f"错误: {error_msg}")
    print(f"错误类型: {type(e).__name__}")
    print(f"错误信息: {str(e)}")
    print("\n完整错误堆栈:")
    print(traceback.format_exc())
    print("="*80 + "\n")

def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='读取Megatron格式的数据集')
    parser.add_argument('-d', '--dataset_prefix', type=str, required=True,
                       help='数据集前缀路径（必需），例如：local_datasets/dataset_name/megatron_dataset')
    parser.add_argument('-m', '--model_name', type=str, required=True,
                       help='模型名称（必需），用于加载tokenizer')
    parser.add_argument('-i', '--index', type=int, default=0,
                       help='要读取的序列索引，默认为0')
    return parser.parse_args()

def main():
    """主函数"""
    try:
        args = parse_arguments()
        
        print(f"数据集前缀: {args.dataset_prefix}")
        
        try:
            # 加载数据集
            dataset = indexed_dataset.IndexedDataset(args.dataset_prefix)
        except Exception as e:
            log_error_with_traceback("加载数据集时发生错误", e)
            return
            
        try:
            # 获取指定索引的序列
            sequence = dataset[args.index]
            print(f"\n序列内容 (索引 {args.index}):")
            print(sequence)
        except Exception as e:
            log_error_with_traceback(f"获取索引 {args.index} 的序列时发生错误", e)
            return
            
        try:
            # 加载tokenizer并解码
            print(f"\n加载tokenizer: {args.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        except Exception as e:
            log_error_with_traceback("加载tokenizer时发生错误", e)
            return
            
        try:
            # 解码并打印文本
            decoded_text = tokenizer.decode(sequence)
            print("\n解码后的文本:")
            print("="*80)
            print(decoded_text)
            print("="*80)
        except Exception as e:
            log_error_with_traceback("解码序列时发生错误", e)
            return
            
    except Exception as e:
        log_error_with_traceback("程序执行过程中发生未预期的错误", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
