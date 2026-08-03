import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "/adapters/triaj-qlora")
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "1") == "1"

SYSTEM_PROMPT = (
    "Sen bir e-ticaret destek ekibinin triyaj asistanısın. "
    "Gelen talebi oku ve SADECE şu anahtarlara sahip geçerli bir JSON döndür: "
    "category, priority, sentiment, needs_human, summary, draft_reply.\n"
    "category: kargo-teslimat | iade-degisim | odeme-fatura | urun-arizasi | "
    "hesap-erisim | kampanya-indirim | diger\n"
    "priority: P1 (para/güvenlik kaybı, yasal risk) | P2 (müşteri engellendi) | "
    "P3 (normal) | P4 (bilgi talebi)\n"
    "sentiment: olumlu | notr | olumsuz | ofkeli\n"
    "needs_human: insana devredilmeli mi (true/false)\n"
    "summary: en fazla 15 kelimelik özet\n"
    "draft_reply: müşteriye gönderilebilecek nazik Türkçe yanıt taslağı"
)

VALID_CATEGORIES = {"kargo-teslimat", "iade-degisim", "odeme-fatura",
                    "urun-arizasi", "hesap-erisim", "kampanya-indirim", "diger"}
VALID_PRIORITIES = {"P1", "P2", "P3", "P4"}

_tok = None
_model = None


def load():
    global _tok, _model
    if _model is not None:
        return _tok, _model

    _tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    kwargs = {"torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
              "device_map": "auto" if torch.cuda.is_available() else None}

    if LOAD_IN_4BIT and torch.cuda.is_available():
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    _model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **kwargs)

    if os.path.isdir(ADAPTER_PATH):
        from peft import PeftModel
        _model = PeftModel.from_pretrained(_model, ADAPTER_PATH)
        print(f"[model] QLoRA adaptörü yüklendi: {ADAPTER_PATH}")
    else:
        print(f"[model] Adaptör yok ({ADAPTER_PATH}), temel model kullanılıyor.")

    _model.eval()
    return _tok, _model


def build_prompt(text: str, channel: str, tier: str) -> str:
    tok, _ = load()
    user = f"Kanal: {channel}\nMüşteri segmenti: {tier}\n\nTalep:\n{text}"
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )


def _coerce(raw: str) -> dict:
    """Model çıktısını şemaya oturt; bozuksa güvenli varsayılana düş."""
    match = re.search(r"\{.*\}", raw, re.S)
    data = {}
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = {}

    cat = data.get("category", "diger")
    pri = data.get("priority", "P3")
    return {
        "category": cat if cat in VALID_CATEGORIES else "diger",
        "priority": pri if pri in VALID_PRIORITIES else "P3",
        "sentiment": data.get("sentiment", "notr"),
        "needs_human": bool(data.get("needs_human", True)),
        "summary": str(data.get("summary", ""))[:200] or "Özet üretilemedi.",
        "draft_reply": str(data.get("draft_reply", ""))[:2000]
        or "Talebinizi aldık, ekibimiz en kısa sürede dönüş yapacak.",
    }


@torch.inference_mode()
def triage(text: str, channel: str = "form", tier: str = "standart") -> dict:
    tok, model = load()
    prompt = build_prompt(text, channel, tier)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    out = model.generate(**inputs, max_new_tokens=320, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    completion = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return _coerce(completion)
