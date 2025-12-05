"""
Qwen3 LoRA Fine-tuning Script
对齐 LLaVA 的参数设计和训练流程
"""

import os
import copy
import pathlib
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import torch
import transformers
from transformers import Trainer, AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset

from dataset import make_supervised_data_module


# 全局 rank 变量
local_rank = None


def rank0_print(*args):
    if local_rank == 0 or local_rank is None:
        print(*args)


@dataclass
class ModelArguments:
    """模型参数（对齐 LLaVA）"""
    model_name_or_path: str = field(default="Qwen/Qwen2.5-7B-Instruct")
    version: Optional[str] = field(default="qwen")


@dataclass
class DataArguments:
    """数据参数（对齐 LLaVA）"""
    data_path: str = field(
        default=None,
        metadata={"help": "Path to the training data (jsonl format)."}
    )
    lazy_preprocess: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):    #TrainingArguments中定义了local_rank这个参数
    """训练参数（对齐 LLaVA，添加 LoRA/QLoRA 支持）"""
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length."},
    )
    
    # QLoRA 量化参数
    double_quant: bool = field( #是否启用双量化，论文的省显存技巧
        default=True,
        metadata={"help": "Compress quantization statistics through double quantization."}
    )
    quant_type: str = field(    #4bit的数据类型
        default="nf4",
        metadata={"help": "Quantization data type: 'fp4' or 'nf4'."}
    )
    bits: int = field(      #4/8出发qloar量化
        default=16,
        metadata={"help": "Number of bits: 16, 8, or 4."}
    )
    
    # LoRA 参数
    lora_enable: bool = field(default=False) #是否启用lora
    lora_r: int = field(default=128) #lora的秩
    lora_alpha: int = field(default=256) #lora的alpha
    lora_dropout: float = field(default=0.05) #lora的dropout
    lora_weight_path: str = field(default="") #lora的权重路径，当前未用到
    lora_bias: str = field(default="none") #lora的bias


def maybe_zero_3(param, ignore_status=False, name=None):
    """处理 DeepSpeed ZeRO-3 参数聚合"""
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, bias):
    """获取 PEFT 状态字典（LoRA 权重）"""
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias.items():
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError
    
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    """获取非 LoRA 的可训练参数"""
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    """查找所有 Linear 层用于 LoRA（排除 lm_head）"""
    cls = torch.nn.Linear
    lora_module_names = set()
    
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """安全保存模型（支持 DeepSpeed）"""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu() for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def train():
    global local_rank
    
    # 解析参数，等价于手动定义一个parser，并且将参数分别添加到3个类
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    
    # 计算精度，将冻结部分的模型使用bf16或者fp16压缩，后续传给bnb做4/8量化时候，指定计算时的dtype
    #权重以4/8的精度加载到显存，但前向和反向计算时使用compute_dtype
    # TODO: 使用4090的时候改成bf16
    compute_dtype = (
        torch.float16 if training_args.fp16 
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    
    # ============ 量化配置（QLoRA）============
    bnb_model_from_pretrained_args = {} #初始化BitsAndBytes量化对应的参数
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            # device_map={"": training_args.device},  #当前rank的gpu, TODO: 不太懂这个
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))
    
    # ============ 加载模型 ============
    rank0_print(f"Loading model from {model_args.model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        **bnb_model_from_pretrained_args
    )
    model.config.use_cache = False      #是否使用缓存，训练的时候设置为False
    
    # ============ QLoRA 预处理 ============
    # 冻结量化权重的梯度，调整ln等层的dtype，保证数值稳定，配合梯度检查点进一步优化。
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        model = prepare_model_for_kbit_training(
            model, 
            use_gradient_checkpointing=training_args.gradient_checkpointing
        )
    
    # ============ Gradient Checkpointing ============
    #一种时间换空间的技术，在前向传播时不保存中间激活值，反向传播需要求导时，再重新计算
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    # ============ LoRA 配置 ============
    # 在模型中插入一些模块，保证量化更稳定
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # ============ 加载 Tokenizer ============
    rank0_print(f"Loading tokenizer from {model_args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        # cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    # Qwen 系列通常需要设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ============ QLoRA 混合精度处理 ============
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    
    # ============ 数据集 ============
    rank0_print(f"Loading dataset from {data_args.data_path}...")
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        model_max_length=training_args.model_max_length
    )
    
    # ============ Trainer ============
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )
    
    # ============ 训练 ============
    rank0_print("*** Starting training ***")
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    trainer.save_state()
    model.config.use_cache = True
    
    # ============ 保存模型 ============
    rank0_print(f"Saving model to {training_args.output_dir}...")
    if training_args.lora_enable:
        # 保存 LoRA 权重
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), 
            training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(
                non_lora_state_dict, 
                os.path.join(training_args.output_dir, 'non_lora_trainables.bin')
            )
    else:
        # 保存全参数模型
        safe_save_model_for_hf_trainer(
            trainer=trainer,
            output_dir=training_args.output_dir
        )
    
    rank0_print("Training completed!")


if __name__ == "__main__":
    train()
