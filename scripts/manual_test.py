import os
import torch
import transformers
from transformers import AutoTokenizer
from src.model.qwen3_ts import Qwen3TSConfig, Qwen3TSForCausalLM
from src.constants import DEFAULT_TS_TOKEN, TS_TOKEN_INDEX

def test_inference():
    # 1. 配置路径
    # 如果要使用 checkpoint-2463，请将 model_path 修改为 checkpoint 的绝对路径或相对路径
    # 例如: model_path = "outputs/pretrain_projector_multi/checkpoint-2463"
    model_path = "Qwen/Qwen3-4B"  # 或者 checkpoint 路径
    # 如果 model_path 是 checkpoint 文件夹，projector_path 可以忽略
    output_dir = "outputs/pretrain_projector_multi" 
    projector_path = os.path.join(output_dir, "mm_projector.bin")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 判断是否加载 checkpoint
    is_checkpoint = os.path.exists(os.path.join(model_path, "config.json"))
    
    # 2. 加载配置
    print(f"Loading config from {model_path}...")
    if is_checkpoint:
        config = Qwen3TSConfig.from_pretrained(model_path)
    else:
        config = Qwen3TSConfig.from_pretrained(model_path)
        # 只有在加载原始 Qwen 模型时才需要手动设置这些参数
        # 如果加载 checkpoint，config.json 中已经包含了这些参数
        config.mm_ts_tower = "patchtst"
        config.ts_d_model = 128
        config.ts_n_layers = 3
        config.ts_n_heads = 16
        config.ts_d_ff = 256
        config.patch_len = 16
        config.stride = 8
        config.context_window = 256
        config.mm_projector_type = "mlp2x_gelu"
        config.use_scale_embedding = True
    
    # 3. 加载模型
    print(f"Loading model from {model_path}...")
    model = Qwen3TSForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device
    )
    
    # 4. 加载训练好的权重 (Projector + Scale Encoder)
    # 只有当 model_path 是原始 Qwen 模型时，才需要额外加载 projector 权重
    # 如果 model_path 是 checkpoint，权重已经包含在模型中了
    if not is_checkpoint and os.path.exists(projector_path):
        print(f"Loading projector weights from {projector_path}")
        model.load_pretrain_projector(projector_path)
    elif not is_checkpoint:
        print(f"Warning: Projector weights not found at {projector_path}")
    else:
        print("Model loaded from checkpoint, skipping separate projector loading.")

    # 5. 加载Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=2048,
        padding_side="right",
        use_fast=False
    )
    # 添加特殊token
    tokenizer.add_tokens([DEFAULT_TS_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    
    # 6. 准备输入
    print("Preparing inputs...")
    text = f"Analyze the following time series: {DEFAULT_TS_TOKEN}\nAnswer:"
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    
    # 构造Dummy Time Series
    # Shape: [batch_size, n_vars, seq_len]
    # 假设单变量，长度256
    seq_len = 256
    n_vars = 1
    timeseries = torch.randn(1, n_vars, seq_len, dtype=model.dtype, device=device)
    
    # 构造Scale Stats (Mean, Std)
    # Shape: [batch_size, n_vars, 2]
    scale_stats = torch.zeros(1, n_vars, 2, dtype=model.dtype, device=device)
    scale_stats[:, :, 1] = 1.0 # Std = 1.0
    
    # 7. 推理
    print("Generating...")
    model.eval()
    with torch.no_grad():
        # prepare_inputs_for_generation 会被 model.generate 调用
        # 我们需要通过 kwargs 传递 timeseries 和 scale_stats
        output_ids = model.generate(
            input_ids,
            timeseries=[timeseries[0]], # List of [n_vars, seq_len]
            scale_stats=[scale_stats[0]], # List of [n_vars, 2]
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            use_cache=True
        )
    
    # 8. 解码
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("-" * 20)
    print("Input:", text)
    print("Output:", output_text)
    print("-" * 20)

if __name__ == "__main__":
    test_inference()
