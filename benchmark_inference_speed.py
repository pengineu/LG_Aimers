"""
benchmark_inference_speed.py
============================
optimum-benchmark Python API를 이용해 양자화된 EXAONE 4.0 1.2B 모델의
토큰당 추론 시간을 측정합니다.

측정 방식
---------
1. [공식 벤치마크] optimum-benchmark InferenceConfig로 랜덤 입력 기반
   latency / throughput 측정 (warmup 포함, 재현 가능한 수치)

2. [실제 데이터 벤치마크] MMLU-Redux 2.0 실제 프롬프트를 사용해
   per-token 추론 시간을 직접 측정 (평균 입력 토큰 수 반영)

사용법
------
  python benchmark_inference_speed.py                  # 기본 (dir 모드)
  python benchmark_inference_speed.py --source zip     # zip 모드
  python benchmark_inference_speed.py --max-samples 50 # MMLU 샘플 수 제한
"""

import argparse
import json
import os
import sys
import time
import zipfile
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# 인자 파싱
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Inference speed benchmark for quantized EXAONE 4.0 1.2B")
    parser.add_argument("--source", choices=["dir", "zip"], default="dir",
                        help="모델 로드 방식: 'dir'=./model 폴더, 'zip'=zip 파일")
    parser.add_argument("--model-dir", default="./model",
                        help="MODEL_SOURCE==dir 일 때 모델 경로")
    parser.add_argument("--zip-path", default="./baseline_submit_12.zip",
                        help="MODEL_SOURCE==zip 일 때 zip 파일 경로")
    parser.add_argument("--zip-subdir", default="model",
                        help="zip 내부 모델 폴더명")

    parser.add_argument("--dataset-id", default="edinburgh-dawg/mmlu-redux-2.0",
                        help="MMLU-Redux 2.0 HuggingFace 데이터셋 ID")
    parser.add_argument("--split", default="test", help="데이터셋 스플릿")
    parser.add_argument("--max-samples", type=int, default=200,
                        help="MMLU 실제 데이터 벤치마크에 사용할 최대 샘플 수 (시간 절약)")

    # optimum-benchmark 공식 벤치마크 설정
    parser.add_argument("--ob-seq-len", type=int, default=128,
                        help="optimum-benchmark 입력 시퀀스 길이")
    parser.add_argument("--ob-new-tokens", type=int, default=1,
                        help="optimum-benchmark generate 최대 새 토큰 수 (1=prefill latency 집중)")
    parser.add_argument("--ob-warmup", type=int, default=5,
                        help="optimum-benchmark 워밍업 횟수")
    parser.add_argument("--ob-runs", type=int, default=10,
                        help="optimum-benchmark 측정 횟수")

    parser.add_argument("--output-dir", default="./benchmark_results",
                        help="결과 저장 폴더")
    parser.add_argument("--skip-ob", action="store_true",
                        help="optimum-benchmark 공식 벤치마크 건너뛰기 (설치 불필요)")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 모델 경로 확인
# ─────────────────────────────────────────────────────────────────────────────
def resolve_model_path(args):
    _tmp_dir = None
    if args.source == "zip":
        print(f"[INFO] {args.zip_path} 압축 해제 중...")
        assert os.path.exists(args.zip_path), f"zip 파일 없음: {args.zip_path}"
        _tmp_dir = tempfile.mkdtemp(prefix="bench_")
        with zipfile.ZipFile(args.zip_path, "r") as zf:
            zf.extractall(_tmp_dir)
        model_path = os.path.join(_tmp_dir, args.zip_subdir)
        print(f"[INFO] 압축 해제 완료: {model_path}")
    else:
        model_path = args.model_dir

    assert os.path.exists(model_path), f"모델 경로 없음: {model_path}"
    print(f"[INFO] 모델 경로: {os.path.abspath(model_path)}")
    return model_path, _tmp_dir


# ─────────────────────────────────────────────────────────────────────────────
# 1. optimum-benchmark 공식 벤치마크
# ─────────────────────────────────────────────────────────────────────────────
def run_optimum_benchmark(model_path: str, args) -> dict:
    """
    optimum-benchmark Python API를 사용해 공식 latency/throughput 측정.
    랜덤 입력 기반이므로 재현 가능하고 표준화된 수치를 제공합니다.
    """
    print("\n" + "="*60)
    print("  [1/2] optimum-benchmark 공식 벤치마크")
    print("="*60)

    try:
        from optimum_benchmark import (
            Benchmark, BenchmarkConfig,
            InferenceConfig, PyTorchConfig,
        )
        from optimum_benchmark.launchers.inline.config import InlineConfig
        from optimum_benchmark.logging_utils import setup_logging
        setup_logging(level="WARNING")
    except ImportError as e:
        print(f"[WARN] optimum-benchmark 미설치 → 건너뜀: {e}")
        print("       pip install optimum-benchmark 으로 설치 후 재실행하세요.")
        return {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    abs_model_path = os.path.abspath(model_path)

    launcher_config = InlineConfig()

    scenario_config = InferenceConfig(
        latency=True,
        memory=True,
        input_shapes={"sequence_length": args.ob_seq_len, "batch_size": 1},
        generate_kwargs={"max_new_tokens": args.ob_new_tokens},
        warmup_runs=args.ob_warmup,
        iteration_runs=args.ob_runs,
    )

    backend_config = PyTorchConfig(
        model=abs_model_path,
        device=device,
        device_ids="0" if device == "cuda" else None,
        torch_dtype="float16" if device == "cuda" else "float32",
        trust_remote_code=True,
    )

    benchmark_config = BenchmarkConfig(
        name="exaone_quantized_inference",
        scenario=scenario_config,
        launcher=launcher_config,
        backend=backend_config,
    )

    print(f"[INFO] 백엔드: PyTorch | 디바이스: {device}")
    print(f"[INFO] 입력 길이: {args.ob_seq_len} tokens | 새 토큰: {args.ob_new_tokens}")
    print(f"[INFO] 워밍업: {args.ob_warmup}회 | 측정: {args.ob_runs}회")
    print("[INFO] 벤치마크 실행 중...")

    try:
        benchmark_report = Benchmark.launch(benchmark_config)

        # 결과 추출
        report_dict = benchmark_report.to_dict() if hasattr(benchmark_report, "to_dict") else {}

        # 주요 지표 추출 시도
        ob_results = {"raw_report": report_dict}

        # latency 파싱 (구조가 버전마다 다를 수 있음)
        try:
            forward = benchmark_report.inference.forward
            ob_results["forward_latency_mean_ms"] = forward.latency.mean * 1000
            ob_results["forward_latency_p50_ms"]  = forward.latency.p50  * 1000
            ob_results["forward_latency_p99_ms"]  = forward.latency.p99  * 1000
            ob_results["forward_throughput_tokens_per_sec"] = forward.throughput.value
        except Exception:
            pass

        try:
            generate = benchmark_report.inference.generate
            ob_results["generate_latency_mean_ms"] = generate.latency.mean * 1000
            ob_results["generate_throughput_tokens_per_sec"] = generate.throughput.value
            # 토큰당 시간 (생성 1토큰 기준)
            ob_results["ms_per_output_token"] = (
                generate.latency.mean * 1000 / max(args.ob_new_tokens, 1)
            )
        except Exception:
            pass

        print("\n[RESULT] optimum-benchmark 결과:")
        for k, v in ob_results.items():
            if k != "raw_report" and isinstance(v, (int, float)):
                print(f"  {k:45s}: {v:.4f}")

        return ob_results

    except Exception as e:
        print(f"[WARN] optimum-benchmark 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 2. MMLU 실제 데이터 기반 추론 속도 측정
# ─────────────────────────────────────────────────────────────────────────────
def build_mmlu_prompt(question: str, choices: list) -> str:
    """benchmark_mmlu.ipynb와 동일한 형식의 프롬프트"""
    CHOICE_TOKENS = ["A", "B", "C", "D"]
    lines = [f"Question: {question}"]
    for letter, choice in zip(CHOICE_TOKENS, choices):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def load_mmlu_samples(args, max_samples: int) -> list:
    """MMLU-Redux 2.0에서 샘플 로드"""
    try:
        from datasets import load_dataset, get_dataset_config_names
    except ImportError:
        print("[ERROR] datasets 패키지 미설치: pip install datasets")
        sys.exit(1)

    print(f"\n[INFO] MMLU-Redux 2.0 로드: {args.dataset_id}")
    records = []

    try:
        configs = get_dataset_config_names(args.dataset_id)
        configs = [c for c in configs if c != "default"]
        print(f"[INFO] config {len(configs)}개 확인")

        for cfg in configs:
            if len(records) >= max_samples:
                break
            try:
                ds = load_dataset(args.dataset_id, cfg, split=args.split, trust_remote_code=True)
                for row in ds:
                    if len(records) >= max_samples:
                        break
                    choices = row.get("choices", row.get("options", []))
                    if isinstance(choices, list) and len(choices) == 4:
                        records.append({
                            "subject": cfg,
                            "question": row["question"],
                            "choices": choices,
                        })
            except Exception:
                continue

    except Exception as e:
        print(f"[WARN] config 조회 실패 → 단일 스플릿 시도: {e}")
        try:
            ds = load_dataset(args.dataset_id, split=args.split, trust_remote_code=True)
            for row in ds:
                if len(records) >= max_samples:
                    break
                choices = row.get("choices", row.get("options", []))
                if isinstance(choices, list) and len(choices) == 4:
                    records.append({
                        "subject": row.get("subject", "unknown"),
                        "question": row["question"],
                        "choices": choices,
                    })
        except Exception as e2:
            print(f"[ERROR] 데이터셋 로드 실패: {e2}")
            return []

    print(f"[INFO] {len(records)}개 샘플 로드 완료")
    return records


def run_mmlu_latency_benchmark(model_path: str, args) -> dict:
    """
    MMLU 실제 프롬프트로 per-token 추론 시간 측정.
    - prefill latency: 프롬프트 입력 처리 시간 (forward pass 1회)
    - 토큰당 시간 = prefill_time / input_token_count (prefill 관점)
    """
    print("\n" + "="*60)
    print("  [2/2] MMLU 실제 데이터 기반 추론 속도 측정")
    print("="*60)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("[ERROR] transformers 미설치")
        return {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    # 토크나이저 & 모델 로드
    print("[INFO] 토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"

    print("[INFO] 모델 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"[INFO] 모델 파라미터: {sum(p.numel() for p in model.parameters())/1e9:.3f}B")

    # MMLU 샘플 로드
    samples = load_mmlu_samples(args, max_samples=args.max_samples)
    if not samples:
        return {}

    # 프롬프트 생성
    prompts = [build_mmlu_prompt(s["question"], s["choices"]) for s in samples]

    # ── 워밍업 (GPU 준비) ────────────────────────────────────────────────────
    WARMUP = 3
    print(f"\n[INFO] 워밍업 {WARMUP}회...")
    with torch.no_grad():
        for p in prompts[:WARMUP]:
            inp = tokenizer(p, return_tensors="pt").to(device)
            _ = model(**inp)
    if device == "cuda":
        torch.cuda.synchronize()

    # ── Prefill Latency 측정 ─────────────────────────────────────────────────
    print(f"[INFO] Prefill latency 측정 중... ({len(prompts)}개 샘플)")
    prefill_times_ms = []
    input_token_counts = []

    with torch.no_grad():
        for prompt in prompts:
            inp = tokenizer(prompt, return_tensors="pt").to(device)
            n_tokens = inp["input_ids"].shape[1]

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(**inp)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            prefill_times_ms.append((t1 - t0) * 1000)
            input_token_counts.append(n_tokens)

    # ── Decode Latency 측정 (generate, max_new_tokens=1) ────────────────────
    print(f"[INFO] Decode latency 측정 중... (max_new_tokens=1)")
    decode_times_ms = []

    with torch.no_grad():
        for prompt in prompts:
            inp = tokenizer(prompt, return_tensors="pt").to(device)

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model.generate(**inp, max_new_tokens=1, do_sample=False)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            decode_times_ms.append((t1 - t0) * 1000)

    # ── 통계 계산 ────────────────────────────────────────────────────────────
    prefill_arr = np.array(prefill_times_ms)
    decode_arr  = np.array(decode_times_ms)
    token_arr   = np.array(input_token_counts)

    # 토큰당 시간 (prefill: input token 기준)
    ms_per_input_token = prefill_arr / token_arr  # shape: (N,)

    # 순수 decode 1토큰 시간 = generate(1 new token) - prefill
    decode_1tok_arr = decode_arr - prefill_arr  # 음수 방지용 clip
    decode_1tok_arr = np.clip(decode_1tok_arr, 0, None)

    results = {
        # Prefill 통계
        "prefill_latency_mean_ms":   float(prefill_arr.mean()),
        "prefill_latency_median_ms": float(np.median(prefill_arr)),
        "prefill_latency_p95_ms":    float(np.percentile(prefill_arr, 95)),
        "prefill_latency_std_ms":    float(prefill_arr.std()),

        # 입력 토큰당 prefill 시간
        "ms_per_input_token_mean":   float(ms_per_input_token.mean()),
        "ms_per_input_token_median": float(np.median(ms_per_input_token)),
        "ms_per_input_token_p95":    float(np.percentile(ms_per_input_token, 95)),

        # Decode 통계 (generate 1 token 포함 전체)
        "decode_total_mean_ms":      float(decode_arr.mean()),
        "decode_total_median_ms":    float(np.median(decode_arr)),
        "decode_1tok_mean_ms":       float(decode_1tok_arr.mean()),
        "decode_1tok_median_ms":     float(np.median(decode_1tok_arr)),

        # 입력 토큰 수 통계
        "input_token_count_mean":   float(token_arr.mean()),
        "input_token_count_median": float(np.median(token_arr)),
        "input_token_count_min":    int(token_arr.min()),
        "input_token_count_max":    int(token_arr.max()),

        # 처리량
        "prefill_throughput_tokens_per_sec": float(
            token_arr.sum() / (prefill_arr.sum() / 1000)
        ),

        # 메타
        "n_samples":  len(prompts),
        "device":     device,
        "dtype":      str(dtype),
    }

    print("\n[RESULT] MMLU 실제 데이터 추론 속도:")
    print(f"  {'항목':<45} {'값':>12}")
    print(f"  {'-'*60}")
    for k, v in results.items():
        if k not in ("device", "dtype", "n_samples"):
            print(f"  {k:<45} {v:>12.4f}")
    print(f"\n  {'n_samples':<45} {results['n_samples']:>12}")
    print(f"  {'device':<45} {results['device']:>12}")
    print(f"  {'dtype':<45} {results['dtype']:>12}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 결과 저장
# ─────────────────────────────────────────────────────────────────────────────
def save_results(ob_results: dict, mmlu_results: dict, model_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    combined = {
        "timestamp":   datetime.now().isoformat(),
        "model_path":  os.path.abspath(model_path),
        "optimum_benchmark": ob_results,
        "mmlu_latency_benchmark": mmlu_results,
    }

    out_path = os.path.join(output_dir, f"inference_speed_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] 결과 저장: {out_path}")

    # 요약 출력
    print("\n" + "█"*60)
    print("   최종 요약: 토큰당 추론 시간")
    print("█"*60)

    if mmlu_results:
        print(f"\n  [MMLU 실제 데이터 기반]")
        print(f"  입력 토큰당 Prefill 시간 (평균):  "
              f"{mmlu_results.get('ms_per_input_token_mean', -1):.4f} ms/token")
        print(f"  입력 토큰당 Prefill 시간 (중앙값): "
              f"{mmlu_results.get('ms_per_input_token_median', -1):.4f} ms/token")
        print(f"  Prefill Throughput:               "
              f"{mmlu_results.get('prefill_throughput_tokens_per_sec', -1):.2f} tokens/sec")
        print(f"  Decode 1 토큰 추가 시간 (평균):   "
              f"{mmlu_results.get('decode_1tok_mean_ms', -1):.4f} ms")

    if ob_results and "ms_per_output_token" in ob_results:
        print(f"\n  [optimum-benchmark 공식]")
        print(f"  토큰당 생성 시간 (generate):  "
              f"{ob_results['ms_per_output_token']:.4f} ms/token")
        if "forward_latency_mean_ms" in ob_results:
            print(f"  Forward (prefill) latency:    "
                  f"{ob_results['forward_latency_mean_ms']:.4f} ms")

    print("█"*60)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print("="*60)
    print("  양자화 모델 추론 속도 벤치마크")
    print("  (optimum-benchmark + MMLU-Redux 2.0)")
    print("="*60)

    # GPU 정보
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("[INFO] GPU 없음, CPU 모드")

    # 모델 경로 확인
    model_path, _tmp_dir = resolve_model_path(args)

    ob_results   = {}
    mmlu_results = {}

    try:
        # 1. optimum-benchmark 공식 벤치마크
        if not args.skip_ob:
            ob_results = run_optimum_benchmark(model_path, args)
        else:
            print("\n[INFO] --skip-ob 옵션: optimum-benchmark 건너뜀")

        # 2. MMLU 실제 데이터 기반 측정
        mmlu_results = run_mmlu_latency_benchmark(model_path, args)

        # 결과 저장
        save_results(ob_results, mmlu_results, model_path, args.output_dir)

    finally:
        if _tmp_dir and os.path.exists(_tmp_dir):
            shutil.rmtree(_tmp_dir)
            print(f"\n[INFO] 임시 디렉토리 삭제: {_tmp_dir}")


if __name__ == "__main__":
    main()
