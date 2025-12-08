"""
多模态Qwen3训练脚本
支持时间序列+文本的两阶段训练：预训练投影层 + LoRA微调
"""

import os
import copy
import pathlib
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import torch
import transformers
from transformers import Trainer, AutoTokenizer
from torch.utils.data import Dataset

# 导入多模态组件
from model.qwen3_ts import Qwen3TSConfig, Qwen3TSForCausalLM
from dataset_multimodal import make_multimodal_data_module
from constants import DEFAULT_TS_TOKEN
import constants as GLOBAL_CONSTANTS

# 全局rank变量
local_rank = None


def rank0_print(*args):
    if local_rank == 0 or local_rank is None:
        print(*args)


@dataclass
class ModelArguments:
    """模型参数"""
    model_name_or_path: str = field(default="Qwen/Qwen3-8B")
    version: Optional[str] = field(default="qwen3_ts")
    
    # 时序编码器配置
    mm_ts_tower: str = field(default="patchtst")
    patchtst_checkpoint: Optional[str] = field(
        default="PatchTST_supervised/checkpoints/checkpoint.pth"
    )
    freeze_patchtst: bool = field(default=True)
    
    # PatchTST超参数
    context_window: int = field(default=256)
    patch_len: int = field(default=16)
    stride: int = field(default=8)
    ts_d_model: int = field(default=128)
    ts_n_layers: int = field(default=3)
    ts_n_heads: int = field(default=16)
    ts_d_ff: int = field(default=256)
    ts_dropout: float = field(default=0.1)
    
    # 投影层配置
    mm_projector_type: str = field(default="mlp2x_gelu")
    
    # 两阶段训练
    tune_mm_mlp_adapter: bool = field(
        default=False,
        metadata={"help": "是否只训练投影层（预训练阶段）"}
    )
    pretrain_mm_mlp_adapter: Optional[str] = field(
        default=None,
        metadata={"help": "预训练投影层权重路径"}
    )


@dataclass
class DataArguments:
    """数据参数"""
    data_path: str = field(
        default=None,
        metadata={"help": "训练数据路径（JSONL格式）"}
    )
    lazy_preprocess: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """训练参数（添加QLoRA/LoRA支持）"""
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    
    model_max_length: int = field(
        default=2048,
        metadata={"help": "最大序列长度"}
    )
    
    # QLoRA量化参数
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=16)
    
    # LoRA参数
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=128)
    lora_alpha: int = field(default=256)
    lora_dropout: float = field(default=0.05)
    lora_weight_path: str = field(default="")
    lora_bias: str = field(default="none")


def maybe_zero_3(param, ignore_status=False, name=None):
    """处理DeepSpeed ZeRO-3参数聚合"""
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
    """获取PEFT状态字典（LoRA权重）"""
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
    """获取非LoRA的可训练参数"""
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    """查找所有Linear层用于LoRA（排除lm_head和mm_projector）"""
    cls = torch.nn.Linear
    lora_module_names = set()
    
    for name, module in model.named_modules():
        if isinstance(module, cls):
            # 排除投影层和lm_head
            if 'mm_projector' in name or 'lm_head' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """安全保存模型（支持DeepSpeed）"""
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
    
    # 解析参数
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    
    # 计算精度
    compute_dtype = (
        torch.float16 if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    
    # ============ 量化配置（QLoRA）============
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
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
    
    # ============ 创建多模态配置 ============
    rank0_print(f"创建多模态Qwen3配置...")
    config = Qwen3TSConfig.from_pretrained(
        model_args.model_name_or_path,
        # 时序编码器配置
        mm_ts_tower=model_args.mm_ts_tower,
        patchtst_checkpoint=model_args.patchtst_checkpoint,
        freeze_patchtst=model_args.freeze_patchtst,
        context_window=model_args.context_window,
        patch_len=model_args.patch_len,
        stride=model_args.stride,
        ts_d_model=model_args.ts_d_model,
        ts_n_layers=model_args.ts_n_layers,
        ts_n_heads=model_args.ts_n_heads,
        ts_d_ff=model_args.ts_d_ff,
        ts_dropout=model_args.ts_dropout,
        # 投影层配置
        mm_projector_type=model_args.mm_projector_type,
        mm_hidden_size=model_args.ts_d_model,
        # 训练配置
        tune_mm_mlp_adapter=model_args.tune_mm_mlp_adapter,
        pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter,
    )
    
    # ============ 加载模型 ============
    rank0_print(f"加载多模态Qwen3模型: {model_args.model_name_or_path}...")
    model = Qwen3TSForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        **bnb_model_from_pretrained_args
    )
    model.config.use_cache = False
    
    rank0_print("✓ 多模态模型加载完成")
    rank0_print(f"  - 时序编码器: {model_args.mm_ts_tower}")
    rank0_print(f"  - 投影层类型: {model_args.mm_projector_type}")
    rank0_print(f"  - PatchTST冻结: {model_args.freeze_patchtst}")
    
    # ============ 参数冻结策略 ============
    if model_args.tune_mm_mlp_adapter:
        # 预训练阶段：只训练投影层（+可选PatchTST）
        rank0_print("\n预训练模式：只训练投影层")
        
        # 冻结Qwen3所有参数
        for p in model.model.parameters():
            p.requires_grad = False
        
        # 解冻投影层
        for p in model.model.mm_projector.parameters():
            p.requires_grad = True
        
        # 解冻scale_encoder（如果存在）
        if hasattr(model.model, 'scale_encoder'):
            for p in model.model.scale_encoder.parameters():
                p.requires_grad = True
            rank0_print("  - 解冻尺度编码器")
        
        # 解冻PatchTST（如果需要）
        if not model_args.freeze_patchtst:
            rank0_print("  - 解冻PatchTST编码器")
            model.get_ts_encoder().unfreeze()
        
        # 打印可训练参数
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        rank0_print(f"  - 可训练参数: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # ============ QLoRA预处理 ============
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
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    # ============ LoRA配置 ============
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        
        rank0_print("\n微调模式：LoRA微调Qwen3")
        
        # 找到所有linear层（排除投影层）
        target_modules = find_all_linear_names(model)
        rank0_print(f"  - LoRA target modules: {target_modules}")
        
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        
        model = get_peft_model(model, lora_config)
        
        # 微调阶段：投影层也参与训练
        for p in model.model.mm_projector.parameters():
            p.requires_grad = True
        
        model.print_trainable_parameters()
    
    # ============ 加载Tokenizer ============
    rank0_print(f"\n加载tokenizer: {model_args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ============ 添加时序特殊token ============
    rank0_print(f"\n添加时序特殊token: {DEFAULT_TS_TOKEN}")
    num_new_tokens = tokenizer.add_tokens([DEFAULT_TS_TOKEN], special_tokens=True)
    
    if num_new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
        
        # 初始化新token的embedding为均值
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        
        rank0_print(f"  - 添加 {num_new_tokens} 个新token")
    
    # 更新全局常量
    GLOBAL_CONSTANTS.TS_TOKEN_INDEX = tokenizer.convert_tokens_to_ids(DEFAULT_TS_TOKEN)
    rank0_print(f"  - TS_TOKEN_INDEX = {GLOBAL_CONSTANTS.TS_TOKEN_INDEX}")
    
    # ============ QLoRA混合精度处理 ============
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
    rank0_print(f"\n加载多模态数据集: {data_args.data_path}...")
    data_module = make_multimodal_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        model_max_length=training_args.model_max_length,
        context_window=model_args.context_window
    )
    
    # ============ Trainer ============
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )
    
    # ============ 训练 ============
    rank0_print("\n" + "="*60)
    rank0_print("开始训练")
    rank0_print("="*60 + "\n")
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    trainer.save_state()
    model.config.use_cache = True
    
    # ============ 保存模型 ============
    rank0_print(f"\n保存模型到: {training_args.output_dir}")
    
    if model_args.tune_mm_mlp_adapter:
        # 预训练阶段：保存投影层和尺度编码器
        rank0_print("  - 保存投影层和尺度编码器权重...")
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            save_dict = {
                'mm_projector': {
                    k: v.cpu() for k, v in model.model.mm_projector.state_dict().items()
                }
            }
            
            # 保存scale_encoder（如果存在）
            if hasattr(model.model, 'scale_encoder'):
                save_dict['scale_encoder'] = {
                    k: v.cpu() for k, v in model.model.scale_encoder.state_dict().items()
                }
                rank0_print(f"  ✓ 尺度编码器权重已保存")
            
            torch.save(
                save_dict,
                os.path.join(training_args.output_dir, 'mm_projector.bin')
            )
            rank0_print(f"  ✓ 投影层权重已保存")
    
    elif training_args.lora_enable:
        # LoRA微调阶段：保存LoRA权重+投影层
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
            rank0_print(f"  ✓ LoRA权重已保存")
            rank0_print(f"  ✓ 投影层权重已保存到non_lora_trainables.bin")
    
    else:
        # 全参数微调：保存完整模型
        safe_save_model_for_hf_trainer(
            trainer=trainer,
            output_dir=training_args.output_dir
        )
        rank0_print(f"  ✓ 完整模型已保存")
    
    rank0_print("\n" + "="*60)
    rank0_print("训练完成！")
    rank0_print("="*60)


if __name__ == "__main__":
    train()
