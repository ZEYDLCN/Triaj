FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv

# GPU'da çalıştıracaksan cu121 tekerleklerini kullan:
#   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "-u", "worker.py"]
