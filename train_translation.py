import torch
import torch.nn as nn
import time
import math
import os
import logging
from tqdm import tqdm


def evaluate(model, eval_dataset):
    """
    Transformer 모델 검증 함수
    Args:
        model (nn.Module): Transformer 모델
        eval_dataset (DataLoader): 검증 데이터셋

    Returns:
        float: 검증 손실 값 (Validation Loss)
    """
    model.eval()  # 모델을 평가 모드로 설정
    total_loss = 0.0
    total_batches = len(eval_dataset)

    with torch.no_grad():  # 그라디언트 계산 비활성화 (메모리 절약)
        for data in eval_dataset:
            input = data['input'].to(model.device)
            target = data['target'].to(model.device)
            input_mask = data['input_mask'].to(model.device)
            target_mask = data['target_mask'].to(model.device)

            _, loss = model(input, target, input_mask, target_mask, labels=target)

            total_loss += loss.item()

    avg_loss = total_loss / total_batches
    return avg_loss


def setup_logger(log_path):
    logger = logging.getLogger("TrainLogger")
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def save_checkpoint(model, optimizer, epoch, global_steps, losses, checkpoint_path, best=False):
    """
    모델 체크포인트 저장 함수
    Args:
        model (nn.Module): Transformer 모델
        optimizer (torch.optim.Optimizer): 옵티마이저
        epoch (int): 현재 학습 Epoch
        global_steps (int): 전체 학습 스텝
        losses (dict): 학습 손실 기록
        checkpoint_path (str): 체크포인트 저장 경로
        best (bool): 최적 모델 저장 여부
    """
    state = {
        'epoch': epoch,
        'train_step': global_steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses
    }

    # 파일명 설정 (최적 모델과 일반 체크포인트 구분)
    filename = "best_model.pth" if best else f"checkpoint_epoch{epoch}_step{global_steps}.pth"
    save_path = os.path.join(checkpoint_path, filename)

    torch.save(state, save_path)
    logging.info(f"✅ Checkpoint saved: {save_path}")

def train(model, epochs, train_dataset, eval_dataset, optimizer, scheduler, save_interval=500, checkpoint_path="./checkpoints", log_path="train.log"):
    """
    Transformer 모델 학습 함수 (체크포인트 저장 기능 추가)
    """
    os.makedirs(checkpoint_path, exist_ok=True)  # 체크포인트 디렉토리 생성
    logger = setup_logger(log_path)

    model.train()
    total_loss = 0.0
    global_steps = 0
    start_time = time.time()
    losses = {}
    best_val_loss = float("inf")

    # 체크포인트 로드
    checkpoint_file = os.path.join(checkpoint_path, "best_model.pth")
    if os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location="cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Checkpoint loaded: {checkpoint_file} (Epoch {checkpoint['epoch']})")

    # 학습 시작
    for epoch in range(epochs):
        epoch_start_time = time.time()
        logger.info(f"===== Start Epoch {epoch} =====")

        pb = tqdm(
            enumerate(train_dataset),
            desc=f'Epoch-{epoch} Iterator',
            total=len(train_dataset),
            bar_format='{l_bar}{bar:10}{r_bar}'
        )

        for i, data in pb:
            # 입력 데이터 로드
            input = data['input'].to(model.device)
            target = data['target'].to(model.device)
            input_mask = data['input_mask'].to(model.device)
            target_mask = data['target_mask'].to(model.device)

            optimizer.zero_grad()
            _, loss = model(input, target, input_mask, target_mask, labels=target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            losses[global_steps] = loss.item()
            total_loss += loss.item()
            global_steps += 1

            # 학습 진행 상황 출력
            if i % save_interval == 0:
                cur_loss = total_loss / save_interval
                elapsed = time.time() - start_time

                log_message = (
                    f'| epoch {epoch:3d} | {i:5d}/{len(train_dataset):5d} batches | '
                    f'lr {scheduler.get_lr()[0]:.4f} | ms/batch {elapsed * 1000 / save_interval:.2f} | '
                    f'loss {cur_loss:.2f} | ppl {math.exp(cur_loss):.2f}'
                )

                pb.set_postfix_str(log_message)
                logger.info(log_message)

                total_loss = 0
                start_time = time.time()

                # ✅ 체크포인트 저장
                save_checkpoint(model, optimizer, epoch, global_steps, losses, checkpoint_path)

        # ✅ 매 Epoch 후 Validation 수행
        val_loss = evaluate(model, eval_dataset)
        model.train()

        logger.info('-' * 89)
        logger.info(f'| End of epoch {epoch:3d} | Time: {time.time() - epoch_start_time:.2f}s | '
                    f'Valid loss {val_loss:.2f} | Valid PPL {math.exp(val_loss):.2f}')
        logger.info('-' * 89)

        # 최적 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, global_steps, losses, checkpoint_path, best=True)

        scheduler.step()

    logger.info("===== Training Completed =====")
