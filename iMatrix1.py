import os
import torch
import shutil
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from compressed_tensors.quantization import QuantizationArgs, QuantizationType, preset_name_to_scheme


# Setting
MODEL_ID = "./base_model"
OUT_DIR  = "./model"

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

# Quantization
SCHEME = "W8A8"
TARGETS = ["Linear"]
IGNORE  = ["embed_tokens", "lm_head", "model.layers.0", "model.layers.1", "model.layers.28", "model.layers.29"]
KV_CACHE_SCHEME = QuantizationArgs(type=QuantizationType("float")) # float8

print("[INFO] 모델 로드 중...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16
)

print("[INFO] 모델/토크나이저 로드 완료")

print("[INFO] 캘리브레이션 데이터 로드 중...")

ds = load_dataset(
    DATASET_ID,
    split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]",
)

def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["conversations"],
            add_generation_prompt=True,
            tokenize=False)
    }

ds = ds.map(preprocess)

print("[INFO] 데이터 전처리 완료")

print(f"[INFO] Quantization 시작 (scheme={SCHEME}, samples={NUM_CALIBRATION_SAMPLES}, max_len={MAX_SEQUENCE_LENGTH})...")

# --- [변경점: imatrix_mse 적용 부분] ---
# 단순 문자열이 아니라, 프리셋 객체로 불러와서 observer를 imatrix_mse로 수정한다.
scheme_obj = preset_name_to_scheme(SCHEME, TARGETS)
scheme_obj.weights.observer = "imatrix_mse"

recipe = [
    # 1. 캘리브레이션 시 중요도(E[x^2]) 수집 (기존 IGNORE 목록 똑같이 적용)
    IMatrixGatherer(ignore=IGNORE),

    # 2. 수집된 중요도를 바탕으로 양자화 수행
    QuantizationModifier(
        config_groups={"group_0": scheme_obj},
        ignore=IGNORE,
        kv_cache_scheme=KV_CACHE_SCHEME,
    )
]
# --------------------------------------

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("[INFO] Quantization 완료")

os.makedirs(OUT_DIR, exist_ok=True)

model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

print(f"[INFO] 모델 저장 완료: {OUT_DIR}")

zip_name = "baseline)imatrix"
print(f"[INFO] {zip_name}.zip 생성 중...")

shutil.make_archive(
    base_name=zip_name,
    format="zip",
    root_dir=".",
    base_dir=OUT_DIR,
)

print(f"[INFO] 생성 완료: {zip_name}.zip")