#!/bin/bash

# ✅ 실행 로그 출력
echo "🚀 Starting training script: run_translation.py"


# ✅ Checkpoints 폴더 생성
CHECKPOINT_DIR="./checkpoints"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    mkdir -p "$CHECKPOINT_DIR"
    echo "✅ Created checkpoints directory: $CHECKPOINT_DIR"
fi

# ✅ 실행 경로 확인
echo "📂 Current Directory: $(pwd)"

# ✅ `run_translation.py` 실행
python run_translation.py

# ✅ 완료 메시지
echo "✅ Training complete!"
