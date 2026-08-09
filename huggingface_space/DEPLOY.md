# QLoRA adaptörünü ücretsiz yayınlama rehberi

İki parça: (1) adaptörü Hugging Face Hub'a yükle, (2) bu klasördeki Space'i
Hugging Face Spaces'e (ücretsiz CPU basic, 16 GB RAM) deploy et.

## 1) Hugging Face hesabı + token

1. https://huggingface.co/join — ücretsiz hesap aç (yoksa).
2. https://huggingface.co/settings/tokens — **New token**, tür: `Write`, kopyala.

## 2) Adaptörü Hub'a yükle

Bunu adaptörün fiziksel olarak durduğu yerde çalıştır (ör. Codespace'inde,
`training/output/triaj-qlora` hangi makinedeyse orada):

```bash
pip install -U huggingface_hub
huggingface-cli login          # 1. adımdaki token'ı yapıştır

python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
REPO = "KULLANICI_ADIN/triaj-qlora"   # kendi HF kullanıcı adınla değiştir
api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
api.upload_folder(folder_path="training/output/triaj-qlora", repo_id=REPO, repo_type="model")
print(f"Yüklendi: https://huggingface.co/{REPO}")
EOF
```

## 3) Space oluştur

1. https://huggingface.co/new-space
2. SDK: **Gradio**, Hardware: **CPU basic (free)**, Visibility: Public.
3. Oluşunca Space'in kendi git adresi verilir, ör.
   `https://huggingface.co/spaces/KULLANICI_ADIN/triaj-demo`.

## 4) Bu klasörü Space'e gönder

```bash
git clone https://huggingface.co/spaces/KULLANICI_ADIN/triaj-demo
cp huggingface_space/app.py huggingface_space/requirements.txt huggingface_space/README.md triaj-demo/
cd triaj-demo
git add -A
git commit -m "Triaj QLoRA demo"
git push
```

(Push sırasında kullanıcı adı + token/parola isteyebilir — token'ı parola
alanına yapıştır.)

## 5) Adaptörü Space'e bağla

Space sayfasında **Settings → Variables and secrets → New variable**:
- `ADAPTER_REPO_ID` = `KULLANICI_ADIN/triaj-qlora` (2. adımdaki repo)

Kaydedince Space otomatik yeniden build alır (~2-3 dk, ilk açılışta model
indirileceği için biraz daha uzun sürebilir). Bittiğinde Space'in üstündeki
"App" sekmesi canlı, herkese açık bir URL verir
(`https://huggingface.co/spaces/KULLANICI_ADIN/triaj-demo`).

## Notlar

- `ADAPTER_REPO_ID` boş bırakılırsa Space yine çalışır, sadece base modelle
  (fine-tune'suz) — kategori/öncelik isabeti düşer ama sistem ayaktadır.
- CPU'da (GPU yok) her sorgu birkaç saniye sürebilir — bu normal, ücretsiz
  katmanın beklenen davranışı.
- Bu Space, ana projedeki Redis/Kong/semantik-cache mimarisini içermez —
  sadece modeli tek başına sorgulamanı sağlar. Tam sistemi görmek için hâlâ
  `docker compose up` (yerelde ya da Codespace'te) gerekir.
