# Triaj — destek talebi sevk panosu

Gelen Türkçe destek taleplerini **kategori + öncelik + duygu + yanıt taslağına** çeviren
küçük bir sistem. QLoRA ile fine-tune edilmiş 1.5B'lik bir model karar verir, Redis
aynı derdi anlatan talepler için modeli hiç çalıştırmaz, Kong da dış dünyaya bakan
tek kapı olur.

```
tarayıcı ──▶ Kong (anahtar + kota + CORS)
                │
                ├─▶ FastAPI ──▶ embed ──▶ Redis vektör araması
                │                 │            ├─ isabet → yanıt anında döner
                │                 │            └─ ıska  → Redis Stream'e yazılır
                │                 └─▶ Pub/Sub ──▶ SSE ──▶ panoya canlı düşer
                └─▶ worker ◀── Stream ── QLoRA modeli ──▶ sonuç + cache'e yazım
```

## Neden bu tasarım

| Katman | İş | Neden |
|---|---|---|
| Kong (DB-less) | API anahtarı, dakikalık/saatlik kota, istek boyutu, CORS, request-id | Kimlik ve kota mantığı uygulama kodundan çıkar; kota sayaçları Redis'te tutulduğu için birden fazla gateway kopyası aynı limiti paylaşır |
| FastAPI | Kabul, doğrulama, embedding, cache araması, SSE | Talebi kabul etmek milisaniyeler sürer, model beklemez |
| Redis Stack | Vektör cache (HNSW/COSINE), Stream kuyruğu, Pub/Sub, sayaçlar, kota | Dört ayrı altyapı yerine tek servis |
| Worker | 4-bit modelle çıkarım | GPU'lu tarafı ayrı ölçeklenir; `docker compose up --scale worker=3` |

**Semantik cache** işin özü: "kargom 9 gündür gelmedi" ile "9 gündür kargo hareket etmiyor"
farklı metinler ama aynı talep. Kosinüs benzerliği 0.93'ü geçerse model hiç çalışmaz —
yanıt ~15 ms'de döner, GPU maliyeti sıfırdır. Panodaki her kartın tepesindeki şerit
o kartın cache'ten mi modelden mi geldiğini gösterir.

## Çalıştırma

```bash
docker compose up -d --build
open http://localhost:8080          # pano
./scripts/smoke.sh                  # uçtan uca test
```

Adaptör yoksa worker temel modelle çalışır; sistem yine ayaktadır, sadece
JSON tutarlılığı ve kategori isabeti düşer.

## Fine-tuning (QLoRA)

GPU'n yoksa Colab'da ücretsiz T4 ile çalıştır: `training/colab_finetune.ipynb`'i
[Colab'da aç](https://colab.research.google.com/github/ZEYDLCN/Triaj/blob/claude/proje-gorevleri-b73j2b/training/colab_finetune.ipynb),
runtime'ı T4 GPU yap ve hücreleri sırayla çalıştır.

Lokalde çalıştırmak istersen:

```bash
cd training
pip install -r requirements.txt
python prepare_dataset.py --n 1500     # sentetik veri (kendi arşivinle değiştir)
python train_qlora.py --epochs 3       # ~15 dk / 16 GB GPU
python evaluate.py                     # temel model vs adaptör
docker compose restart worker          # output/triaj-qlora otomatik bağlanır
```

Yapılandırma: 4-bit NF4 + çift kuantizasyon, LoRA r=16 / alpha=32, tüm attention ve
MLP projeksiyonları hedefli, `paged_adamw_8bit`. Loss yalnızca cevap token'larına
uygulanır (prompt `-100` ile maskelenir) — model prompt'u ezberlemek yerine şemayı
öğrenir. `SYSTEM_PROMPT` tek yerde (`worker/model.py`) tanımlıdır; eğitim ve çıkarım
aynı prompt'u kullanmazsa doğruluk sessizce düşer.

Fine-tuning'in asıl kazancı: 1.5B'lik bir model prompt'la sabit şemayı tutturamaz,
eğitimden sonra tutturur. Yani büyük modele ve uzun few-shot prompt'a gerek kalmaz.

## Uçlar

| Uç | Açıklama |
|---|---|
| `POST /v1/tickets` | Talep gönderir. Cache isabetinde `200` benzeri gövde + `source: cache`, ıskada `status: queued` |
| `GET /v1/tickets/{id}` | Tek talebin sonucu |
| `GET /v1/tickets?limit=50` | Son talepler |
| `GET /events` | SSE canlı akış (`?x-api-key=` ile de anahtar kabul eder — EventSource başlık gönderemez) |
| `GET /v1/stats` | Cache isabet oranı, kuyruk derinliği |
| `GET /metrics` | Prometheus |
| `GET /health` | Anahtar istemez |

## Ayarlanacak yerler

- `CACHE_THRESHOLD` (varsayılan 0.93): düşürürsen daha çok cache isabeti ama yanlış eşleşme riski. Kendi verinde etiketli çiftlerle kalibre et.
- `gateway/kong.yml` içindeki `rate-limiting` kotaları ve `keyauth_credentials` anahtarları.
- `training/prepare_dataset.py`: sentetik şablonlar yerine gerçek ticket arşivi.

## Bilerek yapılmayanlar

Kalıcı veritabanı yok (Redis TTL'i yetiyor), kullanıcı yönetimi yok, çok kiracılı
yapı yok. Bunlar projeyi büyütür ama anlatmak istediği şeye bir şey katmaz.
