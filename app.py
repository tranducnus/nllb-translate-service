"""
NLLB-200 Batch Translation API — CTranslate2 with full CPU thread utilization.

Endpoints:
  GET  /health                      — health check
  POST /translate                   — single text translation
  POST /translate/batch             — batch translation (50+ sentences at once)
  GET  /languages                   — list supported language codes

CTranslate2 config:
  inter_threads=4  — number of parallel translations
  intra_threads=3  — threads per translation (4×3 = 12 = all Ryzen 3600 threads)
  compute_type=int8 — 4x faster than float32 on CPU
"""

import os
import time
import ctranslate2
import sentencepiece as spm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="NLLB-200 Batch API", version="1.0.0")

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model")
INTER_THREADS = int(os.environ.get("INTER_THREADS", "4"))
INTRA_THREADS = int(os.environ.get("INTRA_THREADS", "3"))
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "int8")
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "128"))

translator = None
sp_model = None


@app.on_event("startup")
def load_model():
    global translator, sp_model
    print(f"Loading CTranslate2 model from {MODEL_DIR}")
    print(f"  inter_threads={INTER_THREADS}, intra_threads={INTRA_THREADS}, compute_type={COMPUTE_TYPE}")
    t0 = time.time()
    translator = ctranslate2.Translator(
        MODEL_DIR,
        device="cpu",
        inter_threads=INTER_THREADS,
        intra_threads=INTRA_THREADS,
        compute_type=COMPUTE_TYPE,
    )
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(os.path.join(MODEL_DIR, "sentencepiece.bpe.model"))
    elapsed = time.time() - t0
    print(f"  Model loaded in {elapsed:.1f}s")


def tokenize(text: str, src_lang: str) -> list[str]:
    sp_model.SetEncodeExtraOptions("")
    tokens = sp_model.Encode(text, out_type=str)
    return [src_lang] + tokens


def detokenize(tokens: list[str]) -> str:
    filtered = [t for t in tokens if not t.startswith("__") or not t.endswith("__")]
    return sp_model.Decode(filtered)


def translate_texts(texts: list[str], source: str, target: str, beam_size: int = 4) -> list[dict]:
    source_tokens = [tokenize(t, source) for t in texts]
    target_prefix = [[target]] * len(texts)

    t0 = time.time()
    results = translator.translate_batch(
        source_tokens,
        target_prefix=target_prefix,
        beam_size=beam_size,
        max_decoding_length=512,
        replace_unknowns=True,
    )
    elapsed_ms = (time.time() - t0) * 1000

    translations = []
    for r in results:
        tokens = r.hypotheses[0]
        text = detokenize(tokens)
        translations.append(text)

    return translations, elapsed_ms


class SingleRequest(BaseModel):
    text: str
    source: str = "eng_Latn"
    target: str = "spa_Latn"
    beam_size: int = 4


class BatchRequest(BaseModel):
    texts: list[str]
    source: str = "eng_Latn"
    target: str = "spa_Latn"
    beam_size: int = 4


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": MODEL_DIR,
        "inter_threads": INTER_THREADS,
        "intra_threads": INTRA_THREADS,
        "compute_type": COMPUTE_TYPE,
    }


@app.post("/translate")
def translate_single(req: SingleRequest):
    try:
        translations, elapsed_ms = translate_texts([req.text], req.source, req.target, req.beam_size)
        return {
            "translations": translations,
            "elapsed_ms": round(elapsed_ms, 1),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate/batch")
def translate_batch(req: BatchRequest):
    if len(req.texts) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"Max batch size is {MAX_BATCH_SIZE}")
    if len(req.texts) == 0:
        return {"translations": [], "count": 0, "elapsed_ms": 0}
    try:
        translations, elapsed_ms = translate_texts(req.texts, req.source, req.target, req.beam_size)
        return {
            "translations": translations,
            "count": len(translations),
            "elapsed_ms": round(elapsed_ms, 1),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/languages")
def languages():
    common = [
        {"code": "eng_Latn", "name": "English"},
        {"code": "zho_Hans", "name": "Chinese (Simplified)"},
        {"code": "spa_Latn", "name": "Spanish"},
        {"code": "arb_Arab", "name": "Arabic"},
        {"code": "hin_Deva", "name": "Hindi"},
        {"code": "por_Latn", "name": "Portuguese"},
        {"code": "fra_Latn", "name": "French"},
        {"code": "deu_Latn", "name": "German"},
        {"code": "jpn_Jpan", "name": "Japanese"},
        {"code": "kor_Hang", "name": "Korean"},
        {"code": "rus_Cyrl", "name": "Russian"},
        {"code": "ita_Latn", "name": "Italian"},
        {"code": "nld_Latn", "name": "Dutch"},
        {"code": "tur_Latn", "name": "Turkish"},
        {"code": "pol_Latn", "name": "Polish"},
        {"code": "vie_Latn", "name": "Vietnamese"},
        {"code": "tha_Thai", "name": "Thai"},
        {"code": "ind_Latn", "name": "Indonesian"},
        {"code": "swe_Latn", "name": "Swedish"},
        {"code": "ukr_Cyrl", "name": "Ukrainian"},
        {"code": "zsm_Latn", "name": "Malay"},
    ]
    return common
