# LLM数据处理工具集

本项目包含用于大型语言模型（LLM）数据处理的一系列工具脚本。

## 安装与配置

1. 克隆仓库（包含子模块）:
```bash
git clone --recursive https://github.com/kefoeka/datasets_converter.git
```

2. 如果您是项目创建者，首次添加子模块:
```bash
cd LLM_practicing
git submodule add https://github.com/kefoeka/datasets_converter.git
```

3. 安装依赖:
```bash
pip install -r requirements.txt
```

## 工具说明

### 1. 数据集加载工具 (load_dataset_example.py)

从Hugging Face加载数据集并保存到本地。

#### 用法

```bash
python load_dataset_example.py -d <dataset_name> [-o <output_dir>]
```

#### 参数

- `-d, --dataset_name`: 数据集名称（必需），例如：togethercomputer/RedPajama-Data-1T-Sample
- `-o, --output_dir`: 输出目录路径，默认为 local_datasets

#### 功能

- 从Hugging Face加载指定数据集
- 显示数据集基本信息（分割、特征、大小等）
- 将数据保存为本地JSON文件（每行一个样本）
- 显示保存文件的前5条数据

### 2. Megatron数据转换工具 (megatron_data_converter.py)

将数据集转换为Megatron-LM训练所需的格式。

#### 用法

```bash
python megatron_data_converter.py -i <input_path> -o <output_dir> -m <model_name> [-n <dataset_name>]
```

#### 参数

- `-i, --input_path`: 输入文件路径（必需）
- `-o, --output_dir`: 输出目录路径（必需）
- `-m, --model_name`: 模型名称（必需），例如：AI-ModelScope/Llama-2-70b-hf
- `-n, --dataset_name`: 数据集名称，如果不指定则使用输入文件名（不含后缀）

#### 功能

- 加载JSON格式的文本数据集
- 使用指定的tokenizer对文本进行编码
- 将编码后的数据转换为Megatron-LM索引数据集格式
- 生成.bin和.idx两个文件

### 3. Megatron数据集读取工具 (megatron_dataset_reader.py)

用于读取和验证Megatron格式的数据集。

#### 用法

```bash
python megatron_dataset_reader.py -d <dataset_prefix> -m <model_name> [-i <index>]
```

#### 参数

- `-d, --dataset_prefix`: 数据集前缀路径（必需），例如：local_datasets/dataset_name/megatron_dataset
- `-m, --model_name`: 模型名称（必需），用于加载tokenizer
- `-i, --index`: 要读取的序列索引，默认为0

#### 功能

- 加载Megatron格式的索引数据集
- 读取指定索引的序列内容
- 使用对应的tokenizer解码并显示原始文本

## 完整处理流程示例

1. 下载数据集：
```bash
python load_dataset_example.py -d togethercomputer/RedPajama-Data-1T-Sample
```

2. 转换为Megatron格式：
```bash
python megatron_data_converter.py -i local_datasets/RedPajama-Data-1T-Sample.json -o megatron_data -m AI-ModelScope/Llama-2-70b-hf
```

3. 验证转换后的数据集：
```bash
python megatron_dataset_reader.py -d megatron_data/megatron_dataset -m AI-ModelScope/Llama-2-70b-hf -i 0
``` 