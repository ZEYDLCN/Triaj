import base64
import json
import os
import socket
import time

import redis

import model

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
STREAM = "triaj:stream"
GROUP = "triaj:workers"
EVENTS = "triaj:events"
CACHE_PREFIX = "triaj:cache:"
CACHE_TTL = 60 * 60 * 24 * 7
CONSUMER = f"w-{socket.gethostname()}-{os.getpid()}"

r = redis.Redis.from_url(REDIS_URL, decode_responses=False)


def ensure_group() -> None:
    try:
        r.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
    except redis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


def publish(record: dict) -> None:
    payload = json.dumps(record, ensure_ascii=False)
    r.set(f"triaj:ticket:{record['ticket_id']}", payload, ex=60 * 60 * 24)
    r.zadd("triaj:recent", {record["ticket_id"]: time.time()})
    r.publish(EVENTS, payload)


def handle(fields: dict) -> None:
    g = lambda k, d="": fields.get(k.encode(), d.encode()).decode()
    tid = g("ticket_id")
    text = g("text")
    t0 = time.perf_counter()

    try:
        triaged = model.triage(text, g("channel", "form"), g("customer_tier", "standart"))
        status = "done"
    except Exception as e:                      # model patlarsa talep kaybolmasın
        print(f"[worker] {tid} hata: {e}")
        triaged, status = None, "failed"

    latency = int((time.perf_counter() - t0) * 1000)
    record = {
        "ticket_id": tid, "status": status, "source": "model",
        "latency_ms": latency, "triage": triaged, "text": text,
        "channel": g("channel", "form"), "customer_tier": g("customer_tier", "standart"),
        "created_at": float(g("enqueued_at", "0") or 0),
    }
    publish(record)

    if status == "done":
        r.hincrby("triaj:stats", "model_run", 1)
        # Sonucu semantik cache'e yaz: benzer talep bir daha modele gitmesin
        key = f"{CACHE_PREFIX}{g('cache_key')}"
        r.hset(key, mapping={
            "text": text,
            "category": triaged["category"],
            "payload": json.dumps(triaged, ensure_ascii=False),
            "embedding": base64.b64decode(g("embedding")),
        })
        r.expire(key, CACHE_TTL)

    print(f"[worker] {tid} -> {status} ({latency} ms)")


def main() -> None:
    ensure_group()
    model.load()
    print(f"[worker] {CONSUMER} hazır, kuyruk dinleniyor…")

    while True:
        try:
            batch = r.xreadgroup(GROUP, CONSUMER, {STREAM: ">"}, count=1, block=5000)
        except redis.ConnectionError:
            time.sleep(2)
            continue
        if not batch:
            continue
        for _stream, messages in batch:
            for msg_id, fields in messages:
                try:
                    handle(fields)
                finally:
                    r.xack(STREAM, GROUP, msg_id)


if __name__ == "__main__":
    main()
