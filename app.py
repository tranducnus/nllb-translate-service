"""
NLLB-200 Batch Translation API — CTranslate2 with full CPU thread utilization.

Endpoints:
  GET  /health            — health check
  POST /translate         — single text: { text, source, target }
  POST /translate/batch   — batch: { texts: [...], source, target }

Uses transformers tokenizer for correct NLLB token handling.
"""

import os
import time
import ctranslate2
import transformers
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="NLLB-200 Batch API", version="2.0.0")

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model")
INTER_THREADS = int(os.environ.get("INTER_THREADS", "4"))
INTRA_THREADS = int(os.environ.get("INTRA_THREADS", "3"))
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "int8")
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "128"))

translator = None
tokenizer = None


@app.on_event("startup")
def load_model():
    global translator, tokenizer
    print(f"Loading model from {MODEL_DIR} (inter={INTER_THREADS}, intra={INTRA_THREADS}, {COMPUTE_TYPE})")
    t0 = time.time()
    translator = ctranslate2.Translator(
        MODEL_DIR, device="cpu",
        inter_threads=INTER_THREADS, intra_threads=INTRA_THREADS,
        compute_type=COMPUTE_TYPE,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_DIR, clean_up_tokenization_spaces=True
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")


def do_translate(texts: list[str], source: str, target: str, beam_size: int = 4) -> tuple[list[str], float]:
    tokenizer.src_lang = source
    encoded = [
        tokenizer.convert_ids_to_tokens(tokenizer.encode(t))
        for t in texts
    ]
    target_prefix = [[target]] * len(texts)

    t0 = time.time()
    results = translator.translate_batch(
        encoded,
        target_prefix=target_prefix,
        beam_size=beam_size,
        max_decoding_length=512,
        replace_unknowns=True,
    )
    elapsed_ms = (time.time() - t0) * 1000

    translations = []
    for r in results:
        hyp = r.hypotheses[0][1:]
        hyp = [t for t in hyp if t != "</s>"]
        text = tokenizer.decode(tokenizer.convert_tokens_to_ids(hyp))
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
        translations, elapsed_ms = do_translate([req.text], req.source, req.target, req.beam_size)
        return {"translations": translations, "elapsed_ms": round(elapsed_ms, 1)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate/batch")
def translate_batch(req: BatchRequest):
    if len(req.texts) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"Max batch size is {MAX_BATCH_SIZE}")
    if not req.texts:
        return {"translations": [], "count": 0, "elapsed_ms": 0}
    try:
        translations, elapsed_ms = do_translate(req.texts, req.source, req.target, req.beam_size)
        return {"translations": translations, "count": len(translations), "elapsed_ms": round(elapsed_ms, 1)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
