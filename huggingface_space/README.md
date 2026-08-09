---
title: Triaj QLoRA Demo
emoji: 🎫
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: apache-2.0
---

# Triaj — QLoRA fine-tune demosu

`Qwen2.5-1.5B-Instruct` + Türkçe destek talebi triyajı için eğitilmiş QLoRA adaptörünü
tarayıcıdan ücretsiz sorgulamak için hazırlanmış bir Gradio Space'i.

Ana projeden (`ZEYDLCN/Triaj`) bağımsız çalışır — Redis/Kong/worker gerektirmez,
sadece modeli host eder. Kurulum adımları için repo kökündeki
`huggingface_space/DEPLOY.md` dosyasına bak.
