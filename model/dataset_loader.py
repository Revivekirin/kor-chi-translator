import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, dataset_path, split="train", max_length=64, tokenizer=None):
        """
        한국어-중국어 번역 데이터 로드 (CSV 기반)
        Args:
            dataset_path (str): 데이터셋 ZIP 파일이 압축된 폴더 경로
            split (str): "train" 또는 "validation"
            max_length (int): 최대 문장 길이 (패딩 기준)
            tokenizer (callable): 텍스트를 토큰 ID로 변환하는 함수 (예: SentencePiece, BERT tokenizer)
        """
        self.max_length = max_length
        self.tokenizer = tokenizer

        split_dir = "Training" if split == "train" else "Validation"
        data_dir = os.path.join(dataset_path, split_dir)

        self.file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")]

        self.data = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in self.file_list], ignore_index=True)

        self.data = self.data[["한국어", "중국어"]].dropna()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text = self.data.iloc[idx]["한국어"]
        tgt_text = self.data.iloc[idx]["중국어"]

        if self.tokenizer:
            src_tokenized = self.tokenizer(src_text, max_length=self.max_length, padding="max_length", truncation=True)
            tgt_tokenized = self.tokenizer(tgt_text, max_length=self.max_length, padding="max_length", truncation=True)

            src_ids = src_tokenized["input_ids"]
            tgt_ids = tgt_tokenized["input_ids"]

            input_mask = src_tokenized["attention_mask"]
            target_mask = tgt_tokenized["attention_mask"]
        else:
            src_ids = [ord(ch) for ch in src_text][:self.max_length]  # 예제 (char-level 토큰화)
            tgt_ids = [ord(ch) for ch in tgt_text][:self.max_length]

            input_mask = [1] * len(src_ids) + [0] * (self.max_length - len(src_ids)) 
            target_mask = [1] * len(tgt_ids) + [0] * (self.max_length - len(tgt_ids))

        return {
            "input": torch.tensor(src_ids, dtype=torch.long),
            "target": torch.tensor(tgt_ids, dtype=torch.long),
            "input_mask": torch.tensor(input_mask, dtype=torch.long),
            "target_mask": torch.tensor(target_mask, dtype=torch.long)
        }
