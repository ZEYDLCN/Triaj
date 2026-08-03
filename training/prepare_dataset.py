"""Sentetik Türkçe destek talebi veri seti üretir.

Kullanım:
    python prepare_dataset.py --n 1200 --out data/

Gerçek projede burayı kendi ticket arşivinle değiştir; şablonlar sadece
QLoRA eğitiminin uçtan uca çalıştığını görmen için var.
"""
import argparse
import json
import pathlib
import random

random.seed(7)

TEMPLATES = [
    # (kategori, öncelik, duygu, insana devir, şablonlar)
    ("kargo-teslimat", "P3", "olumsuz", False, [
        "{gun} gündür kargom hareket etmiyor, takip numarası {takip}. Ne zaman gelecek?",
        "Sipariş {siparis} teslim edildi görünüyor ama elime ulaşmadı.",
        "Kargo şubesi adresi bulamamış, yeniden gönderim yapabilir misiniz?",
    ]),
    ("kargo-teslimat", "P4", "notr", False, [
        "{siparis} numaralı siparişimin tahmini teslim tarihi nedir?",
        "Kargo firmasını değiştirebilir miyim?",
    ]),
    ("iade-degisim", "P3", "notr", False, [
        "{urun} ürününü bedeni büyük geldiği için iade etmek istiyorum.",
        "Değişim talebim {gun} gündür onaylanmadı, süreç ne kadar sürüyor?",
    ]),
    ("iade-degisim", "P2", "ofkeli", True, [
        "İade ettim, ürün deponuza ulaştı ama {gun} gündür param yatmadı. Tüketici hakemine gideceğim.",
        "İadem reddedilmiş, gerekçe bile yazılmamış. Bu kabul edilemez.",
    ]),
    ("odeme-fatura", "P1", "ofkeli", True, [
        "Kartımdan {tutar} TL iki kez çekildi, acilen iade edin.",
        "Hesabımdan hiç sipariş vermediğim halde {tutar} TL çekilmiş, dolandırıcılık olabilir.",
    ]),
    ("odeme-fatura", "P3", "notr", False, [
        "{siparis} numaralı siparişin e-faturasını gönderir misiniz?",
        "Fatura adresimi kurumsal olarak güncellemek istiyorum.",
    ]),
    ("urun-arizasi", "P2", "olumsuz", True, [
        "{urun} kutudan arızalı çıktı, açılmıyor bile.",
        "{urun} {gun} gün kullandıktan sonra bozuldu, garanti kapsamında mı?",
    ]),
    ("hesap-erisim", "P2", "olumsuz", False, [
        "Hesabıma giriş yapamıyorum, şifre sıfırlama e-postası gelmiyor.",
        "Telefon numaramı değiştirdim, doğrulama kodu eski numaraya gidiyor.",
    ]),
    ("hesap-erisim", "P1", "ofkeli", True, [
        "Hesabım ele geçirilmiş, adresim değiştirilmiş ve sipariş verilmiş. Acil kapatın.",
    ]),
    ("kampanya-indirim", "P4", "notr", False, [
        "{kod} indirim kodu sepette çalışmıyor, neden?",
        "Kargo bedava kampanyası hâlâ geçerli mi?",
    ]),
    ("kampanya-indirim", "P3", "olumlu", False, [
        "Kampanyayı kaçırdım ama ürün çok iyi, tekrar indirime girecek mi?",
    ]),
    ("diger", "P4", "olumlu", False, [
        "Destek ekibiniz çok hızlı dönüş yaptı, teşekkür etmek istedim.",
        "Mağazalarınızın çalışma saatlerini öğrenebilir miyim?",
    ]),
]

URUNLER = ["kablosuz kulaklık", "kahve makinesi", "koşu ayakkabısı", "robot süpürge",
           "akıllı saat", "mont", "klavye", "blender"]
KODLAR = ["YAZ25", "ILKALIS", "HOSGELDIN10", "KARGOBEDAVA"]
KANALLAR = ["email", "chat", "form", "telefon"]
TIERS = ["standart", "plus", "kurumsal"]

REPLIES = {
    "kargo-teslimat": "Merhaba, kargo kaydınızı inceledik. Gönderiyi kargo firmasıyla takibe aldık ve en geç 24 saat içinde size dönüş yapacağız.",
    "iade-degisim": "Merhaba, iade talebinizi öncelikli sıraya aldık. Ürün depomuza ulaştıktan sonra ödeme iadeniz 3 iş günü içinde tamamlanacaktır.",
    "odeme-fatura": "Merhaba, ödeme kaydınızı finans ekibimize acil olarak ilettik. Fazla çekilen tutar en geç 3 iş günü içinde kartınıza iade edilecektir.",
    "urun-arizasi": "Merhaba, ürünü garanti kapsamında değerlendirmeye alıyoruz. Adresinizden ücretsiz kargo ile teslim alınması için sizinle iletişime geçeceğiz.",
    "hesap-erisim": "Merhaba, hesap güvenliğiniz için erişimi geçici olarak kısıtladık. Kimlik doğrulaması sonrası hesabınızı birlikte yeniden açacağız.",
    "kampanya-indirim": "Merhaba, kampanya koşullarını kontrol ettik. Kodun geçerli olduğu ürün grubunu ve son kullanım tarihini aşağıda paylaşıyoruz.",
    "diger": "Merhaba, mesajınız için teşekkür ederiz. Sorunuzu ilgili ekibe ilettik, kısa süre içinde bilgilendirileceksiniz.",
}


def make_row() -> dict:
    cat, pri, sent, human, temps = random.choice(TEMPLATES)
    text = random.choice(temps).format(
        gun=random.randint(2, 21),
        takip=random.randint(10**9, 10**10 - 1),
        siparis=f"SP-{random.randint(100000, 999999)}",
        urun=random.choice(URUNLER),
        tutar=random.choice([249, 599, 1249, 2899, 7499]),
        kod=random.choice(KODLAR),
    )
    if random.random() < 0.3:
        text += random.choice([" Lütfen acele edin.", " Bilgi verirseniz sevinirim.",
                               " Yanıtınızı bekliyorum.", ""])
    return {
        "text": text,
        "channel": random.choice(KANALLAR),
        "customer_tier": random.choice(TIERS),
        "label": {
            "category": cat, "priority": pri, "sentiment": sent,
            "needs_human": human,
            "summary": f"{cat.replace('-', ' ')} konulu {pri} öncelikli talep",
            "draft_reply": REPLIES[cat],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    rows, seen = [], set()
    while len(rows) < args.n:
        row = make_row()
        if row["text"] in seen:
            continue
        seen.add(row["text"])
        rows.append(row)

    random.shuffle(rows)
    split = int(len(rows) * 0.9)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name, chunk in (("train", rows[:split]), ("val", rows[split:])):
        path = out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in chunk:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{path}: {len(chunk)} örnek")


if __name__ == "__main__":
    main()
