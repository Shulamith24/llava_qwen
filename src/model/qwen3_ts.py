"""
多模态Qwen3模型
结合PatchTST时序编码器和Qwen3语言模型，实现时间序列+文本的多模态问答
"""

import os
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from transformers import Qwen3Config, Qwen3Model, Qwen3ForCausalLM

from .ts_encoder import PatchTSTEncoderWrapper, build_ts_encoder
from .projector import build_projector
from ..constants import IGNORE_INDEX, DEFAULT_TS_TOKEN
from .. import constants


class Qwen3TSConfig(Qwen3Config):
    """
    多模态Qwen3配置类
    
    扩展Qwen3Config，添加时序编码器和投影层配置
    """
    model_type = "qwen3_ts"
    
    def __init__(
        self,
        # 时序编码器配置
        mm_ts_tower: str = "patchtst",
        patchtst_checkpoint: Optional[str] = None,
        freeze_patchtst: bool = True,
        #patchtst超参
        context_window: int = 256,
        patch_len: int = 16,
        stride: int = 8,
        ts_d_model: int = 128,
        ts_n_layers: int = 3,
        ts_n_heads: int = 16,
        ts_d_ff: int = 256,
        ts_dropout: float = 0.1,
        
        # 投影层配置
        mm_projector_type: str = "mlp2x_gelu",
        mm_hidden_size: Optional[int] = None,  # 如果None，使用ts_d_model
        
        # 训练配置
        tune_mm_mlp_adapter: bool = False,  # 是否只训练投影层（预训练阶段）
        pretrain_mm_mlp_adapter: Optional[str] = None,  # 预训练投影层权重路径
        
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # 时序编码器
        self.mm_ts_tower = mm_ts_tower
        self.patchtst_checkpoint = patchtst_checkpoint
        self.freeze_patchtst = freeze_patchtst
        self.context_window = context_window
        self.patch_len = patch_len
        self.stride = stride
        self.ts_d_model = ts_d_model
        self.ts_n_layers = ts_n_layers
        self.ts_n_heads = ts_n_heads
        self.ts_d_ff = ts_d_ff
        self.ts_dropout = ts_dropout
        
        # 投影层
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = mm_hidden_size or ts_d_model
        
        # 训练
        self.tune_mm_mlp_adapter = tune_mm_mlp_adapter
        self.pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter  #预训练投影层权重路径


class Qwen3TSModel(Qwen3Model):
    """
    继承Qwen3Model，添加时序编码器和投影层
    在Backbone中添加时序编码器和适配器
    """
    config_class = Qwen3TSConfig
    
    def __init__(self, config: Qwen3TSConfig):
        super().__init__(config)
        
        # 时序编码器
        if hasattr(config, 'mm_ts_tower') and config.mm_ts_tower:
            self.ts_encoder = build_ts_encoder(
                checkpoint_path=config.patchtst_checkpoint,
                freeze=config.freeze_patchtst,
                context_window=config.context_window,
                patch_len=config.patch_len,
                stride=config.stride,
                d_model=config.ts_d_model,
                n_layers=config.ts_n_layers,
                n_heads=config.ts_n_heads,
                d_ff=config.ts_d_ff,
                dropout=config.ts_dropout
            )
            
            # MLP投影层
            self.mm_projector = build_projector(
                projector_type=config.mm_projector_type,
                input_dim=config.mm_hidden_size,
                output_dim=config.hidden_size
            )
    
    def get_ts_encoder(self):
        """获取时序编码器"""
        ts_encoder = getattr(self, 'ts_encoder', None)
        if isinstance(ts_encoder, list):
            ts_encoder = ts_encoder[0]
        return ts_encoder
    
    def encode_timeseries(self, time_series_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        编码时间序列（批量，逐样本处理）
        
        Args:
            time_series_list: List of [n_vars, seq_len]
            
        Returns:
            features_list: List of [n_vars * n_patches, hidden_size]
            
        Note:
            Flatten顺序: [var0的所有patches, var1的所有patches, ...]
            这样第i个<ts> token可以直接对应var_i的特征
        """
        ts_encoder = self.get_ts_encoder()
        if ts_encoder is None:
            raise ValueError("时序编码器未初始化")
        
        features_list = []
        for ts in time_series_list:
            # ts: [n_vars, seq_len]
            # 编码为 [n_vars, n_patches, d_model]
            ts_features = ts_encoder(ts)
            
            # Flatten按变量顺序: [n_vars * n_patches, d_model]
            # 顺序: [var0_patch0, ..., var0_patchN, var1_patch0, ..., var1_patchN, ...]
            # 这样第i个<ts>对应的特征就是 [i*n_patches : (i+1)*n_patches]
            n_vars, n_patches, d_model = ts_features.shape
            ts_features = ts_features.reshape(n_vars * n_patches, d_model)
            
            # 投影到Qwen3空间: [n_vars * n_patches, hidden_size]
            projected = self.mm_projector(ts_features)
            
            features_list.append(projected)
        
        return features_list


class Qwen3TSForCausalLM(Qwen3ForCausalLM):
    """
    多模态Qwen3因果语言模型
    
    用于时间序列+文本的多模态问答任务
    
    架构:
    1. 时序编码器(PatchTST): 时间序列 -> 特征向量
    2. 投影层(MLP): 时序特征 -> Qwen3 embedding空间
    3. Qwen3模型: 融合后的embedding -> 文本生成
    
    使用方式:
    - timeseries: List of [n_vars, seq_len]
    - input_ids: 包含<ts> token作为占位符
    """
    config_class = Qwen3TSConfig
    
    def __init__(self, config: Qwen3TSConfig):
        # 会调用Qwen3ForCausalLM的父类PreTrainedModel.init,从而跳过Qwen3ForCausalLM的init，避免创建模型
        # TODO: 看一下源码的实现
        super(Qwen3ForCausalLM, self).__init__(config)
        
        # 使用多模态Model
        self.model = Qwen3TSModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) #TODO:能成功加载hf权重吗？
        
        # PreTrainedModel提供的hook,负责权重初始化
        self.post_init()
        
        # 加载预训练投影层权重
        if hasattr(config, 'pretrain_mm_mlp_adapter') and config.pretrain_mm_mlp_adapter:
            self.load_pretrain_projector(config.pretrain_mm_mlp_adapter)
    
    def get_model(self):
        return self.model
    
    def get_ts_encoder(self):
        return self.model.get_ts_encoder()
    
    #加载预训练投影层权重
    def load_pretrain_projector(self, projector_path: str):
        """加载预训练投影层权重"""
        if not os.path.exists(projector_path):
            print(f"警告：投影层权重不存在: {projector_path}")
            return
        
        print(f"从 {projector_path} 加载投影层权重...")
        weights = torch.load(projector_path, map_location='cpu')
        
        # 提取mm_projector权重
        if 'mm_projector' in weights:
            projector_weights = weights['mm_projector']
        else:
            projector_weights = {k.replace('mm_projector.', ''): v 
                                for k, v in weights.items() if 'mm_projector' in k}
        
        self.model.mm_projector.load_state_dict(projector_weights, strict=False)
        print("投影层权重加载成功")
    
    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[List[torch.FloatTensor]],
        labels: Optional[torch.LongTensor],
        timeseries: Optional[List[torch.Tensor]]
    ):
        """
        准备多模态输入
        
        将<ts> token位置替换为时序特征embedding
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            past_key_values: KV cache
            labels: [batch, seq_len]
            timeseries: List of [n_vars, seq_len]，长度为batch_size
            
        Returns:
            None, attention_mask, past_key_values, inputs_embeds, labels
        """
        ts_encoder = self.get_ts_encoder()
        
        # 如果没有时序数据或编码器，直接返回
        if ts_encoder is None or timeseries is None or input_ids.shape[1] == 1:
            if past_key_values is not None and ts_encoder is not None and \
               timeseries is not None and input_ids.shape[1] == 1:
                # 生成阶段的attention mask调整
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
            return input_ids, attention_mask, past_key_values, None, labels
        
        # 编码时间序列
        ts_features_list = self.model.encode_timeseries(timeseries)
        # ts_features_list: List of [n_vars * n_patches, hidden_size]
        
        # 构建融合的embedding序列
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # 当前样本的时序特征
            cur_ts_features = ts_features_list[batch_idx]  # [n_vars * n_patches, hidden_size]
            
            # 找到<ts> token的位置
            ts_token_indices = torch.where(cur_input_ids == constants.TS_TOKEN_INDEX)[0]
            num_ts_tokens = len(ts_token_indices)
            
            if num_ts_tokens == 0:
                # 没有时序数据的样本
                cur_input_embeds = self.model.embed_tokens(cur_input_ids)
                # 添加dummy feature以保持梯度流动
                cur_input_embeds = cur_input_embeds + \
                    (0. * self.model.mm_projector(ts_encoder.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                continue
            
            # 计算每个<ts>对应的特征数量
            # cur_ts_features: [n_vars * n_patches, hidden_size]
            # 每个<ts>对应一个变量的所有patches
            total_ts_length = cur_ts_features.shape[0]
            tokens_per_ts = total_ts_length // num_ts_tokens
            
            # 验证：确保能整除
            if total_ts_length % num_ts_tokens != 0:
                raise ValueError(
                    f"时序特征总长度({total_ts_length})无法被<ts>数量({num_ts_tokens})整除！"
                    f"这可能意味着编码器输出的特征数量与预期不符。"
                )
            
            # 构建新的embedding序列
            cur_new_input_embeds = []
            cur_new_labels = [] if labels is not None else None
            if labels is not None:
                cur_labels = labels[batch_idx]
                assert cur_labels.shape == cur_input_ids.shape
            
            # 逐个处理<ts> token
            # 第i个<ts>对应第i个变量的所有patch特征
            last_idx = 0
            for ts_idx, ts_token_pos in enumerate(ts_token_indices):
                # <ts>之前的文本embedding
                if ts_token_pos > last_idx:
                    cur_new_input_embeds.append(
                        self.model.embed_tokens(cur_input_ids[last_idx:ts_token_pos])
                    )
                    if labels is not None:
                        cur_new_labels.append(cur_labels[last_idx:ts_token_pos])
                
                # 替换<ts>为对应变量的时序特征
                # 第ts_idx个<ts> -> 第ts_idx个变量的n_patches个特征
                start_idx = ts_idx * tokens_per_ts
                end_idx = (ts_idx + 1) * tokens_per_ts
                ts_feature_chunk = cur_ts_features[start_idx:end_idx]  # [n_patches, hidden_size]
                cur_new_input_embeds.append(ts_feature_chunk)
                
                # 时序token位置的label设为IGNORE_INDEX
                if labels is not None:
                    cur_new_labels.append(
                        torch.full(
                            (ts_feature_chunk.shape[0],),
                            IGNORE_INDEX,
                            device=labels.device,
                            dtype=labels.dtype
                        )
                    )
                
                last_idx = ts_token_pos + 1
            
            # <ts>之后剩余的文本
            if last_idx < len(cur_input_ids):
                cur_new_input_embeds.append(
                    self.model.embed_tokens(cur_input_ids[last_idx:])
                )
                if labels is not None:
                    cur_new_labels.append(cur_labels[last_idx:])
            
            # 拼接embedding序列
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)
        
        # 对齐长度（padding）
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)
            
            # Padding embeddings
            new_input_embeds_aligned = []
            for cur_embed in new_input_embeds:
                pad_len = max_len - cur_embed.shape[0]
                if pad_len > 0:
                    cur_embed = torch.cat([
                        cur_embed,
                        torch.zeros(
                            (pad_len, cur_embed.shape[1]),
                            dtype=cur_embed.dtype,
                            device=cur_embed.device
                        )
                    ], dim=0)
                new_input_embeds_aligned.append(cur_embed)
            new_input_embeds = torch.stack(new_input_embeds_aligned, dim=0)
            
            # Padding labels
            if labels is not None:
                new_labels_aligned = []
                for cur_label in new_labels:
                    pad_len = max_len - cur_label.shape[0]
                    if pad_len > 0:
                        cur_label = torch.cat([
                            cur_label,
                            torch.full(
                                (pad_len,),
                                IGNORE_INDEX,
                                dtype=cur_label.dtype,
                                device=cur_label.device
                            )
                        ], dim=0)
                    new_labels_aligned.append(cur_label)
                new_labels = torch.stack(new_labels_aligned, dim=0)
            
            # 更新attention mask
            if attention_mask is not None:
                new_attention_mask = []
                for cur_attn, cur_label in zip(attention_mask, new_labels):
                    # 新增的部分设为True，padding部分设为False
                    pad_left = cur_label.shape[0] - labels.shape[1]
                    pad_right = max_len - cur_label.shape[0]
                    
                    new_attn = torch.cat([
                        torch.ones(pad_left, dtype=cur_attn.dtype, device=cur_attn.device),
                        cur_attn,
                        torch.zeros(pad_right, dtype=cur_attn.dtype, device=cur_attn.device)
                    ], dim=0)
                    new_attention_mask.append(new_attn)
                attention_mask = torch.stack(new_attention_mask, dim=0)
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)
            
            if attention_mask is not None:
                pad_left = new_input_embeds.shape[1] - input_ids.shape[1]
                attention_mask = torch.cat([
                    torch.ones(
                        (attention_mask.shape[0], pad_left),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    ),
                    attention_mask
                ], dim=1)
        
        return None, attention_mask, past_key_values, new_input_embeds, new_labels
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,     # [batch, ids_length]
        attention_mask: Optional[torch.Tensor] = None, # [batch, ids_length], padding部分为False TODO:作用？
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None, 
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        timeseries: Optional[List[torch.Tensor]] = None,  # 关键参数
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        前向传播
        
        Args:
            input_ids: [batch, seq_len] 
            attention_mask: [batch, seq_len]    #用来告知模型哪些位置是padding，哪些位置是真实的token
            position_ids: [batch, seq_len]      #从attention_mask生成，非pad+1，pad+0
            past_key_values: KV cache           #初次调用
            inputs_embeds: [batch, seq_len, hidden_size]
            labels: [batch, seq_len]
            use_cache: 是否使用KV cache
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出隐藏状态
            timeseries: List of [n_vars, seq_len]，时间序列数据
            return_dict: 是否返回字典
            
        Returns:
            CausalLMOutputWithPast
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 准备多模态输入

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = \
            self.prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, timeseries
            )
        
        # Qwen3 forward
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        # 计算loss
        loss = None
        if labels is not None:
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        """准备生成阶段的输入"""
        # KV cache处理
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            
            input_ids = input_ids[:, remove_prefix_length:]
        
        # Position IDs
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        
        # 构建model inputs
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        
        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "timeseries": kwargs.get("timeseries", None),
        })
        
        return model_inputs


# 注册配置和模型
AutoConfig.register("qwen3_ts", Qwen3TSConfig)
AutoModelForCausalLM.register(Qwen3TSConfig, Qwen3TSForCausalLM)
