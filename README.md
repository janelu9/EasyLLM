# *EasyLLM*

Training Large Language Model faster, easily and low-cost. 

✦ Both GPU and NPU are supported.

✦ Directly training on whole big data write by Spark when pretrain.

✦ Flash speed when fine-tuning because of  no redundant computation .

✦ Make PCIE as fast as NVlinks under 20 billion level model.

✦ Minimalist implementation of Sequence Parallelism (4D Parallelism for extra long context).

✦ High performance of RLHF.

## Installation

```shell
pip wheel -e . --no-deps && pip install jllm-*-py3-none-any.whl
```

## Data Processing

This step is optional but recommended especially when your data are too big to be loaded to CPU memory at once.

### Conversion

Convert the raw data to token ids stored in parquet files.

```shell
python -m jllm.raw2ids \
    --tokenizer Qwen2.5-7B-Instruct \
    -i dataset0.jsonl \
    -o dataset0_Qwen2.5-7B-Instruct
```

Pre-train dataset's samples should be separated by *`'\n\n'`* in text files or be the value of  key*`'text'`* in jsonl files. Fine-tune's format should be *`[{'system':content},{'user':content},{'assistant':content},...] `* in each row of jsonl files, key*`'system'`* is not necessary. RLHF's format is like *`[index,{'user':content}] `.*  *`index`* is an ID of integer.

**For Vision Language Model:**

```shell
python -m jllm.raw2ids \
    --tokenizer Qwen2.5-VL-7B-Instruct \
    -i dataset_vl.jsonl \
    --image_path images \
    --max_len 32769 \
```

Folder *`images`* stores all the images data.  Format of  *`dataset_vl.jsonl`* is like:

*`[{'user':['Give a description of these pictures please.\n <image>....','image0.jpg',...]},{'assistant':'This is ....'}]`*

### Shuffle (For Pretrain)

If you have multiple datasets, you shouldn't skip this step. It could shuffle all the datasets globally by rows like [Spark](https://spark.apache.org) doing. 

Firstly, move all the datasets stored in parquet folders into one directory. such as `datasets`:

```shell
datasets
├── dataset0_Qwen2.5-7B-Instruct
│   ├── dataset0-00000
│   │   ├── dataset0-00000-00000.gzip.parquet
│   │   └── dataset0-00000-00001.gzip.parquet
│   └── dataset0-00001
│       ├── dataset0-00001-00000.gzip.parquet
│       └── dataset0-00001-00001.gzip.parquet
└── dataset1_Qwen2.5-7B-Instruct
    ├── dataset1-00000
    │   ├── dataset1-00000-00000.gzip.parquet
    │   └── dataset1-00000-00001.gzip.parquet
    └── dataset1-00001
        ├── dataset1-00001-00000.gzip.parquet
        └── dataset1-00001-00001.gzip.parquet
```

Then run the following command to shuffle the rows inner each dataset and distribute them to new blocks, `num_block` is recommended to be the multiple of next step's repartition number.

```shell
python -m jllm.shuffle_datasets -d datasets -o shuffled_datasets -n 4
```

Every dataset would be shuffled and placed in `shuffled_datasets` with several times of `num_block` parquet files:

```shell
shuffled_datasets/
├── dataset0_Qwen2.5-7B-Instruct-00000
│   ├── dataset0_Qwen2.5-7B-Instruct-00000-00000.gzip.parquet
│   ├── dataset0_Qwen2.5-7B-Instruct-00000-00001.gzip.parquet
│   ├── dataset0_Qwen2.5-7B-Instruct-00000-00002.gzip.parquet
│   └── dataset0_Qwen2.5-7B-Instruct-00000-00003.gzip.parquet
└── dataset1_Qwen2.5-7B-Instruct-00000
    ├── dataset1_Qwen2.5-7B-Instruct-00000-00000.gzip.parquet
    ├── dataset1_Qwen2.5-7B-Instruct-00000-00001.gzip.parquet
    ├── dataset1_Qwen2.5-7B-Instruct-00000-00002.gzip.parquet
    └── dataset1_Qwen2.5-7B-Instruct-00000-00003.gzip.parquet
```

### Repartition (For Pretrain)

Optional but recommended. 1B token ids in parquet files take up to 2G of hard disk at most but require approximately 10G of CPU memory. Setting `num_partition` according to the CPU memory of each worker.

```shell
python -m jllm.repartition -d shuffled_datasets -n 4
```

The datasets will be:

```shell
shuffled_datasets/
├── 5984729befe338e6a7-part-00000
│   ├── dataset0_Qwen2.5-7B-Instruct-00000-00000.gzip.parquet
│   └── dataset1_Qwen2.5-7B-Instruct-00000-00000.gzip.parquet
├── 5984729befe338e6a7-part-00001
│   ├── dataset0_Qwen2.5-7B-Instruct-00000-00001.gzip.parquet
│   └── dataset1_Qwen2.5-7B-Instruct-00000-00001.gzip.parquet
├── 5984729befe338e6a7-part-00002
│   ├── dataset0_Qwen2.5-7B-Instruct-00000-00002.gzip.parquet
│   └── dataset1_Qwen2.5-7B-Instruct-00000-00002.gzip.parquet
├── 5984729befe338e6a7-part-00003
│   ├── dataset0_Qwen2.5-7B-Instruct-00000-00003.gzip.parquet
│   └── dataset1_Qwen2.5-7B-Instruct-00000-00003.gzip.parquet
└── data.info
```

*Note: You can also use **PySpark** to do these steps. jllm could directly read token ids from the parquets those write out by **Spark** .* 

## Model Training

#### Large Language Model (4D Parallelism):

```shell
deepspeed --module jllm.train_pipe \
    --model Qwen2.5-7B-Instruct \
    --num_train_epochs 3 \
    --train_data dataset0_Qwen2.5-7B-Instruct \
    --pipe_parallel_size 2 \
    --tensor_parallel_size 1 \
    --sequence_parallel_size 2 \
    --per_device_train_batch_size 1 \
    --global_batch_size 32 \
    --partition_method fast \
    --split_dlayer \
    --only_ckpt_model \
    --max_num_checkpoints 2 \
    --learning_rate 1e-5 \
    --checkpoint checkpoint
```

#### **Vision Language Model**:

```shell
deepspeed -H $HOSTFILE \
    --module jllm.train_pipe \
    --model Qwen2.5-VL-7B-Instruct \
    --num_train_epochs 3 \
    --train_data dataset_vl_Qwen2.5-VL-7B-Instruct \
    --pipe_parallel_size 4 \
    --tensor_parallel_size 4 \
    --encoder_pipe_parallel_size 2 \
    --per_device_train_batch_size 1 \
    --global_batch_size 64 \
    --only_ckpt_model \
    --max_num_checkpoints 2 \
    --partition_method fast \
    --split_dlayer \
    --no_pin_memory \
    --checkpoint_grad_interval 1 \
    --checkpoint checkpoint
```

***Note**: Arguments `train_data` and `eval_data` also support `jsonl` file. Run `python -m jllm.train_pipe -h ` for more arguments.* 

Generally, every GPU process reads one piece of data, that means one node with 8 GPUs will need to allocate a total of 8x CPU memory for data.  But now they need just 1x if these GPUs belong to one pipeline under my special optimizations in this project . **I strongly recommend you to train your model with faster and low-cost Pipeline Parallelism** rather than ZERO. Pipeline engine could directly load and save model's weights in HuggingFace's format. It could also load weights from checkpoint. If you want to resume interruption, any configs related to training shouldn't be modified. 

The engine was designed to save checkpoint through background process by default to save more time for training. **Don't save checkpoint too frequently** unless you disable checkpoint in background via the argument '`--background_executor none`' to avoid out of CPU memory.

Setting `--partition_method` to be `fast` will always get a faster training when GPU memory are enough.

#### **Reinforcement Learning** (GRPO):

1. Define a reward function in a python file which should include a `reward_func`:

```python
# reward.py
def reward_func(index,response):
    '''
    Args:
        index: int
            Unique index of the train sample.
        response: Tensor (n,) 
            One generated token ids of trained actor.
    return:
    	score: Tensor (1,)
    		The reward sorce of this response.
    '''
    '''TODO'''
    return score
```

2. Start a ray cluster for vLLM.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ray start --head --port 6380
```

3. Train:

```shell
deepspeed --module jllm.train_pipe \
    --model Qwen2.5-7B-Instruct \
    --num_train_epochs 3 \
    --train_data dataset0_Qwen2.5-7B-Instruct \
    --pipe_parallel_size 1 \
    --tensor_parallel_size 8 \
    --per_device_train_batch_size 1 \
    --global_batch_size 32 \
    --partition_method fast \
    --split_dlayer \
    --only_ckpt_model \
    --max_num_checkpoints 2 \
    --learning_rate 1e-5 \
    --checkpoint checkpoint \
    --rlhf \
    --num_generations 16 \
    --max_new_tokens 2048 \
    --vllm_tp 8 \
    --ray_gpus 8 \
    --vllm_mem 0.12 \
    --reward_func reward.py
```

#### Checkpoint Conversion

If argument `--only_ckpt_model`  is enabled , engine will directly only checkpoint model's weights with HF's format.

You can also convert model's weights from deepspeed's checkpoint to HF's format by `jllm.train_pipe`, such as:

```shell
deepspeed -H $HOSTFILE \
    --module jllm.train_pipe \
    --model Qwen2-VL-7B-Instruct \
    --train_data dataset_vl_Qwen2-VL-7B-Instruct \
    --pipe_parallel_size 8 \
    --encoder_pipe_parallel_size 2 \
    --partition_method 11,2 \
    --split_dlayer \
    --num_train_epochs 0 \
    --from_ckpt checkpoint --tag 1000 \
    --output_dir output_path
```

To concatenate the weights when ` model_parallel_size>1`:

```shell
python -m jllm.cat2hf \
       -C checkpoint_model \
       -H huggingface_model
```

#### Supported Models

|                       Model                        | Training Speed (tokens/s) |
| :------------------------------------------------: | :-----------------------: |
| deepseek-v3-671b (includes multi-token prediction) |                           |
|                     qwen2.5-vl                     |                           |
|                      qwen2-vl                      |             -             |
|                     internvl2                      |             -             |
|                     internlm2                      |             -             |
|                     qwen2-moe                      |             -             |
|                      qwen-14b                      |       80749.57(old)       |
|                    baichuan-13b                    |       79765.50(old)       |
|                     llama-13b                      |       92749.82(old)       |

***Note**: The training speed of each model was measured on 64 NVIDIA A100-PCIE-40GB GPUs linked by 100Gb/s bandwidth of InfiniBand with data type of bfloat16 and batch token size of 2048\*2048 (batch_size\*sequence_length,  batch_size = micro_batch_size \* gradient_accumulation_steps).*

|  Model   | Training Speed (tokens/s) |
| :------: | :-----------------------: |
| llama-7b |         26335.232         |

*8 NVIDIA A100-PCIE-40GB GPUs,  bfloat16, 2304\*2048 tokens/batch.*

|    Model    | Training Speed (tokens/s) |
| :---------: | :-----------------------: |
| Qwen2.5-72b |         125327.23         |

*512 **Ascend-910B-64GB NPUs** of Air-cooled, bfloat16, 4096\*4096 tokens/batch.*

## Citation

If you find EasyLLM useful or use EasyLLM's code  in your research, please cite it in your publications.

```bibtex
@misc{EasyLLM,
  author       = {Jian Lu},
  title        = {EasyLLM: Training Large Language Model faster, easily and low-cost.},
  year         = {2023},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/janelu9/EasyLLM.git}},
}
```

## Acknowledgment

This repository benefits from [DeepSpeed](https://github.com/microsoft/DeepSpeed),  [Megatron-LM](https://github.com/NVIDIA/Megatron-LM.git) and [Flash-Attention](https://github.com/Dao-AILab/flash-attention.git).
