import sys
root_path = "/root/data1/qwen3_finetune"
sys.path.append(root_path)
from transformers import AutoTokenizer
from src.dataset import SupervisedDataset

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
dataset = SupervisedDataset(
    data_path="/root/data1/datasets/ChatTS/align_256/train.jsonl",
    tokenizer=tokenizer,
    model_max_length=2048
)


sample = dataset[0]
print(sample['input_ids'].shape)
print(sample.keys())