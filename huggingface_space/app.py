"""Triaj QLoRA adaptörünü Hugging Face Spaces'in ücretsiz CPU katmanında
(16 GB RAM) sorgulanabilir hale getiren tek dosyalık Gradio uygulaması.

Ana projedeki worker/model.py ile AYNI mantığı (SYSTEM_PROMPT, _coerce) taşır —
Space kendi git deposu olduğu için ana repodan import edilemez, bu yüzden
kasıtlı olarak kopyalanmıştır. worker/model.py değişirse burayı da güncelle.

Ortam değişkenleri (Space > Settings > Variables and secrets):
  ADAPTER_REPO_ID  — HF Hub'a yüklediğin adaptör deposu, ör. "kullaniciadi/triaj-qlora".
                     Boş bırakılırsa base model adaptörsüz çalışır (yine de JSON üretir,
                     ama kategori/öncelik isabeti fine-tune'suz haliyle düşer).
  BASE_MODEL       — varsayılan Qwen/Qwen2.5-1.5B-Instruct
"""
import json
import os
import re

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_REPO_ID = os.getenv("ADAPTER_REPO_ID", "").strip()

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
    _model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)

    if ADAPTER_REPO_ID:
        from peft import PeftModel
        _model = PeftModel.from_pretrained(_model, ADAPTER_REPO_ID)
        print(f"[model] QLoRA adaptörü yüklendi: {ADAPTER_REPO_ID}")
    else:
        print("[model] ADAPTER_REPO_ID ayarlı değil, base model kullanılıyor.")

    _model.eval()
    return _tok, _model


def _coerce(raw: str) -> dict:
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
def triage(text: str, channel: str, tier: str) -> dict:
    if not text.strip():
        return {"error": "Boş talep gönderilemez."}

    tok, model = load()
    user = f"Kanal: {channel}\nMüşteri segmenti: {tier}\n\nTalep:\n{text}"
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
    out = model.generate(**inputs, max_new_tokens=320, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    completion = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return _coerce(completion)


with gr.Blocks(title="Triaj — QLoRA Demo") as demo:
    gr.Markdown(
        "# Triaj — QLoRA fine-tune demosu\n"
        "Türkçe bir destek talebi yaz, model kategori/öncelik/duygu/yanıt taslağı üretsin. "
        + ("Adaptör yüklü: `" + ADAPTER_REPO_ID + "`" if ADAPTER_REPO_ID
           else "⚠️ `ADAPTER_REPO_ID` ayarlanmamış, base model (fine-tune'suz) çalışıyor.")
    )
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="Talep metni", lines=4,
                              placeholder="9 gündür kargom gelmedi, takip numarası...")
            channel = gr.Dropdown(["email", "chat", "form", "telefon"], value="form", label="Kanal")
            tier = gr.Dropdown(["standart", "plus", "kurumsal"], value="standart", label="Müşteri segmenti")
            btn = gr.Button("Triyaj et", variant="primary")
        with gr.Column():
            out = gr.JSON(label="Sonuç")

    btn.click(triage, inputs=[text, channel, tier], outputs=out)
    gr.Examples(
        examples=[
            ["9 gündür kargom hareket etmiyor, takip numarası 1234567890.", "form", "standart"],
            ["Kartımdan aynı tutar iki kez çekildi, acilen iade edin!", "chat", "plus"],
            ["Destek ekibiniz çok hızlı dönüş yaptı, teşekkürler.", "email", "standart"],
        ],
        inputs=[text, channel, tier],
    )

if __name__ == "__main__":
    demo.queue().launch()
