#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Megatron数据转换工具
将数据集转换为Megatron-LM训练所需的格式
"""

# 标准库导入
import os
import sys
import time
import random
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# 第三方库导入
import numpy as np
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer
import sentencepiece as spm

# 本地导入
MEGATRON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Megatron-LM")
sys.path.append(MEGATRON_PATH)
from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

# 配置常量
DEFAULT_SEED = 42
DEFAULT_DTYPE = np.int32
MAX_ERRORS = 100
SHOW_TEXT_INTERVAL = 60  # 显示文本的间隔（秒）
SHOW_PROGRESS_INTERVAL = 5  # 更新进度条的间隔（秒）
TEXT_PREVIEW_LENGTH = 100  # 文本预览长度

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def setup_random_seed(seed: int = DEFAULT_SEED) -> None:
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='将数据集转换为Megatron格式')
    parser.add_argument('-i', '--input_path', type=str, required=True,
                       help='输入文件路径（必需）')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                       help='输出目录路径（必需）')
    parser.add_argument('-n', '--dataset_name', type=str,
                       help='数据集名称，如果不指定则使用输入文件名（不含后缀）')
    parser.add_argument('-m', '--model_name', type=str, required=True,
                       help='模型名称（必需），例如：AI-ModelScope/Llama-2-70b-hf')
    return parser.parse_args()

def get_file_paths(args: argparse.Namespace, current_dir: Path) -> Tuple[Path, Path]:
    """获取输入输出文件路径"""
    input_file = Path(args.input_path)
    output_dir = Path(args.output_dir)
    
    # 如果未指定数据集名称，则使用输入文件名（不含后缀）
    if not args.dataset_name:
        args.dataset_name = input_file.stem
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    return input_file, output_dir

def load_tokenizer(model_name: str) -> Any:
    """加载tokenizer并处理可能的错误"""
    logger.info(f"正在加载tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("设置pad_token为eos_token")
        return tokenizer
    except Exception as e:
        logger.error(f"Tokenizer加载错误，正在检查tokenizer.model文件...")
        try:
            bpe_file = f"{model_name}/tokenizer.model"
            logger.error(f"尝试读取tokenizer.model文件: {bpe_file}")
            
            if os.path.exists(bpe_file):
                sp = spm.SentencePieceProcessor()
                sp.Load(bpe_file)
                logger.error(f"词汇表大小: {sp.GetPieceSize()}")
                logger.error("前10个token:")
                for i in range(min(10, sp.GetPieceSize())):
                    logger.error(f"ID: {i}, Token: {sp.IdToPiece(i)}")
            else:
                logger.error(f"文件不存在: {bpe_file}")
        except Exception as debug_e:
            logger.error(f"调试过程中发生错误: {str(debug_e)}")
        raise

def process_dataset(text_dataset: Dataset, tokenizer: Any, output_dir: Path) -> Tuple[int, int]:
    """处理数据集并转换为Megatron格式"""
    total_tokens = 0
    error_count = 0
    last_show_text_time = 0
    last_show_progress_time = 0
    
    # 创建输出文件
    dataset_prefix = output_dir / "megatron_dataset"
    bin_file = str(dataset_prefix) + ".bin"
    idx_file = str(dataset_prefix) + ".idx"
    
    try:
        builder = IndexedDatasetBuilder(bin_file, dtype=DEFAULT_DTYPE)
        logger.info(f"创建IndexedDatasetBuilder成功，使用数据类型: {DEFAULT_DTYPE.__name__}")
    except Exception as e:
        logger.error(f"创建IndexedDatasetBuilder时发生错误: {str(e)}")
        return total_tokens, error_count
    
    # 使用tqdm进度条处理数据集
    with tqdm(text_dataset, desc="处理样本", unit="样本", ncols=100, 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as pbar:
        for i, example in enumerate(pbar):
            try:
                if not isinstance(example["text"], str) or not example["text"].strip():
                    pbar.write(f"跳过样本 {i}: 文本为空或不是字符串")
                    continue
                
                token_ids = tokenizer(example["text"], return_tensors="pt")["input_ids"][0]
                if len(token_ids) == 0:
                    pbar.write(f"跳过样本 {i}: 分词后长度为0")
                    continue
                
                builder.add_item(token_ids)
                total_tokens += len(token_ids)
                
                # 更新进度显示
                current_time = time.time()
                if current_time - last_show_progress_time >= SHOW_PROGRESS_INTERVAL:
                    formatted_tokens = f"{total_tokens:,}"
                    formatted_avg = f"{total_tokens//(i + 1):,}" if i > 0 else "0"
                    pbar.set_postfix({
                        "总词元": formatted_tokens,
                        "错误": error_count,
                        "平均": formatted_avg
                    })
                    last_show_progress_time = current_time
                
                # 显示文本预览
                if current_time - last_show_text_time >= SHOW_TEXT_INTERVAL:
                    text_preview = (example["text"][:TEXT_PREVIEW_LENGTH] + "..."
                                  if len(example["text"]) > TEXT_PREVIEW_LENGTH
                                  else example["text"])
                    pbar.write("\n" + "="*80)
                    pbar.write(f"正在处理文本 (样本 {i+1}/{len(text_dataset)}):")
                    pbar.write(text_preview)
                    pbar.write("="*80 + "\n")
                    last_show_text_time = current_time
                    
            except Exception as e:
                pbar.write(f"处理样本 {i} 时发生错误: {str(e)}")
                error_count += 1
                if error_count > MAX_ERRORS:
                    pbar.write("错误数量过多，终止处理")
                    break
    
    # 完成数据集构建
    try:
        builder.end_document()
        builder.finalize(idx_file)
        return total_tokens, error_count
    except Exception as e:
        logger.error(f"完成数据集构建时发生错误: {str(e)}")
        return total_tokens, error_count

def main() -> None:
    """主函数"""
    try:
        setup_random_seed()
        args = parse_arguments()
        current_dir = Path(__file__).parent.absolute()
        
        # 获取文件路径
        input_file, output_dir = get_file_paths(args, current_dir)
        logger.info(f"输入文件: {input_file}")
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"模型名称: {args.model_name}")
        
        # 检查输入文件
        if not input_file.exists():
            logger.error(f"错误: 文件 '{input_file}' 不存在!")
            return
        
        # 加载数据集
        try:
            dataset = Dataset.from_json(str(input_file))
        except Exception as e:
            logger.error(f"加载数据集时发生错误: {str(e)}")
            return
        
        if "text" not in dataset.features:
            logger.error(f"错误: 数据集中没有'text'字段。可用字段: {list(dataset.features.keys())}")
            return
        
        # 处理数据集
        text_dataset = dataset.select_columns(["text"])
        logger.info(f"数据集加载完成，共有{len(text_dataset)}条文本")
        
        # 加载tokenizer
        try:
            tokenizer = load_tokenizer(args.model_name)
        except Exception as e:
            logger.error(f"加载tokenizer时发生错误: {str(e)}")
            return
        
        # 转换数据集
        total_tokens, error_count = process_dataset(text_dataset, tokenizer, output_dir)
        
        # 显示最终结果
        logger.info("\n" + "="*80)
        logger.info("数据转换完成！")
        logger.info(f"共处理了 {len(text_dataset)-error_count:,} 个样本")
        logger.info(f"总计 {total_tokens:,} 个词元")
        logger.info(f"平均每个样本 {total_tokens//(len(text_dataset)-error_count):,} 个词元" 
                   if (len(text_dataset)-error_count) > 0 else "平均每个样本 0 个词元")
        logger.info(f"输出文件: {output_dir}/megatron_dataset.bin 和 {output_dir}/megatron_dataset.idx")
        if error_count > 0:
            logger.warning(f"处理过程中跳过了 {error_count:,} 个有问题的样本")
        logger.info("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"程序执行过程中发生未预期的错误: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()