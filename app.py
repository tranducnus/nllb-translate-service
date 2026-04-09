"""
NLLB-200 Translation API
CPU-optimized service using CTranslate2 int8 quantization.
Runs on Hetzner AX41 (AMD Ryzen 5 3600, 64GB RAM, no GPU).

Model: facebook/nllb-200-distilled-600M (CTranslate2 int8)
~350MB RAM footprint, ~0.5-2s per sentence on CPU.
"""

import os
import time
import ctranslate2
import transformers
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="NLLB-200 Translation API",
    description="Self-hosted translation service using Meta's NLLB-200 (600M distilled, int8)",
    version="1.0.0",
)

MODEL_DIR = os.environ.get("NLLB_MODEL_DIR", "/app/model")
INTER_THREADS = int(os.environ.get("NLLB_INTER_THREADS", "4"))
INTRA_THREADS = int(os.environ.get("NLLB_INTRA_THREADS", "2"))

translator = None
tokenizer = None


# ============================================================================
# NLLB-200 language codes (FLORES-200 format)
# Maps ISO 639-1 shortcodes to NLLB's internal codes for the 20 most popular
# languages by global internet usage / news readership.
# ============================================================================
LANG_CODE_MAP = {
    "en": "eng_Latn",
    "zh": "zho_Hans",    # Simplified Chinese
    "es": "spa_Latn",
    "ar": "arb_Arab",    # Modern Standard Arabic
    "hi": "hin_Deva",
    "pt": "por_Latn",    # Brazilian Portuguese
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ru": "rus_Cyrl",
    "it": "ita_Latn",
    "nl": "nld_Latn",
    "tr": "tur_Latn",
    "pl": "pol_Latn",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "id": "ind_Latn",    # Indonesian
    "sv": "swe_Latn",
    "uk": "ukr_Cyrl",
    "ms": "zsm_Latn",    # Malay (Standard)
}


def resolve_lang_code(code: str) -> str:
    """Resolve ISO 639-1 or FLORES-200 code to NLLB internal format."""
    if code in LANG_CODE_MAP:
        return LANG_CODE_MAP[code]
    if "_" in code and len(code) == 8:
        return code
    raise ValueError(
        f"Unknown language code: {code}. "
        f"Use ISO 639-1 ({', '.join(LANG_CODE_MAP.keys())}) or FLORES-200 format (e.g. eng_Latn)."
    )


@app.on_event("startup")
def load_model():
    global translator, tokenizer
    translator = ctranslate2.Translator(
        MODEL_DIR,
        device="cpu",
        compute_type="int8",
        inter_threads=INTER_THREADS,
        intra_threads=INTRA_THREADS,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_DIR, src_lang="eng_Latn"
    )


# ============================================================================
# Request / Response models
# ============================================================================

class TranslateRequest(BaseModel):
    text: str | list[str] = Field(..., description="Text or list of texts to translate")
    source: str = Field("en", description="Source language (ISO 639-1 or FLORES-200)")
    target: str = Field(..., description="Target language (ISO 639-1 or FLORES-200)")
    beam_size: int = Field(4, ge=1, le=10)
    max_length: int = Field(512, ge=1, le=2048)


class TranslateResponse(BaseModel):
    translations: list[str]
    source: str
    target: str
    elapsed_ms: float


class BatchTranslateRequest(BaseModel):
    texts: list[str] = Field(..., description="List of texts to translate", max_length=100)
    source: str = Field("en", description="Source language")
    target: str = Field(..., description="Target language")
    beam_size: int = Field(4, ge=1, le=10)
    max_length: int = Field(512, ge=1, le=2048)


class BatchTranslateResponse(BaseModel):
    translations: list[str]
    source: str
    target: str
    count: int
    elapsed_ms: float


class LanguageInfo(BaseModel):
    iso_code: str
    nllb_code: str
    name: str


LANGUAGE_NAMES = {
    "en": "English", "zh": "Chinese", "es": "Spanish", "ar": "Arabic",
    "hi": "Hindi", "pt": "Portuguese", "fr": "French", "de": "German",
    "ja": "Japanese", "ko": "Korean", "ru": "Russian", "it": "Italian",
    "nl": "Dutch", "tr": "Turkish", "pl": "Polish", "vi": "Vietnamese",
    "th": "Thai", "id": "Indonesian", "sv": "Swedish", "uk": "Ukrainian",
    "ms": "Malay",
}


# ============================================================================
# Core translation function
# ============================================================================

def translate_texts(
    texts: list[str],
    src_lang: str,
    tgt_lang: str,
    beam_size: int = 4,
    max_length: int = 512,
) -> list[str]:
    src_code = resolve_lang_code(src_lang)
    tgt_code = resolve_lang_code(tgt_lang)

    tokenizer.src_lang = src_code

    all_tokens = []
    for text in texts:
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
        all_tokens.append(tokens)

    results = translator.translate_batch(
        all_tokens,
        target_prefix=[[tgt_code]] * len(all_tokens),
        beam_size=beam_size,
        max_decoding_length=max_length,
        repetition_penalty=1.2,
    )

    translated = []
    for result in results:
        output_tokens = result.hypotheses[0]
        if tgt_code in output_tokens:
            output_tokens = [t for t in output_tokens if t != tgt_code]
        decoded = tokenizer.decode(tokenizer.convert_tokens_to_ids(output_tokens))
        translated.append(decoded)

    return translated


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "nllb-200-distilled-600M-ct2-int8",
        "device": "cpu",
    }


@app.get("/languages", response_model=list[LanguageInfo])
def list_languages():
    return [
        LanguageInfo(iso_code=iso, nllb_code=nllb, name=LANGUAGE_NAMES.get(iso, iso))
        for iso, nllb in LANG_CODE_MAP.items()
    ]


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    texts = req.text if isinstance(req.text, list) else [req.text]
    if not texts:
        raise HTTPException(status_code=400, detail="No text provided")

    start = time.monotonic()
    try:
        translations = translate_texts(texts, req.source, req.target, req.beam_size, req.max_length)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {e}")
    elapsed_ms = (time.monotonic() - start) * 1000

    return TranslateResponse(
        translations=translations,
        source=req.source,
        target=req.target,
        elapsed_ms=round(elapsed_ms, 1),
    )


@app.post("/translate/batch", response_model=BatchTranslateResponse)
def translate_batch(req: BatchTranslateRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    if len(req.texts) > 100:
        raise HTTPException(status_code=400, detail="Max 100 texts per batch")

    start = time.monotonic()
    try:
        translations = translate_texts(req.texts, req.source, req.target, req.beam_size, req.max_length)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {e}")
    elapsed_ms = (time.monotonic() - start) * 1000

    return BatchTranslateResponse(
        translations=translations,
        source=req.source,
        target=req.target,
        count=len(translations),
        elapsed_ms=round(elapsed_ms, 1),
    )
