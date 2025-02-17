import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from train_translation import train
from model.transformer import Transformer  
from model.dataset_loader import TranslationDataset  


def collate_fn(batch):
    input_tensors = [item["input"] for item in batch]
    target_tensors = [item["target"] for item in batch]
    input_mask_tensors = [item["input_mask"] for item in batch]  
    target_mask_tensors = [item["target_mask"] for item in batch]  

    input_padded = torch.nn.utils.rnn.pad_sequence(input_tensors, batch_first=True, padding_value=0)
    target_padded = torch.nn.utils.rnn.pad_sequence(target_tensors, batch_first=True, padding_value=0)
    input_mask_padded = torch.nn.utils.rnn.pad_sequence(input_mask_tensors, batch_first=True, padding_value=0)  # ✅ 추가
    target_mask_padded = torch.nn.utils.rnn.pad_sequence(target_mask_tensors, batch_first=True, padding_value=0)  # ✅ 추가

    return {
        "input": input_padded,
        "target": target_padded,
        "input_mask": input_mask_padded, 
        "target_mask": target_mask_padded  
    }


DATASET_PATH = "/home/sophia435256/workspace2/dataset/korea_china_word_bag/korea_china_word_bag"
CHECKPOINT_PATH = "./checkpoints"

model_name = 'transformer-translation-spoken'
vocab_num = 22000
max_length = 64
d_model = 512
head_num = 8
dropout = 0.1
N = 6
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

epochs = 50
batch_size = 8
learning_rate = 0.8
save_interval = 500  


train_dataset = TranslationDataset(DATASET_PATH, split="train")
eval_dataset = TranslationDataset(DATASET_PATH, split="validation")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


model = Transformer(
    vocab_num=vocab_num,
    dim=d_model,
    max_seq_len=max_length,
    head_num=head_num,
    dropout=dropout,
    N=N
).to(device)


optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)  # 학습률 감소


train(
    model=model, 
    epochs=epochs,
    train_dataset=train_loader,
    eval_dataset=eval_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    save_interval=save_interval,
    log_path="training.log"
)

print("🚀 Training complete!")
