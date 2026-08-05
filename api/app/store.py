import base64
import json
import time
from typing import Optional

import numpy as np
import redis.asyncio as redis
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from .config import Keys, settings

pool = redis.ConnectionPool.from_url(settings.redis_url, decode_responses=False)


def client() -> redis.Redis:
    return redis.Redis(connection_pool=pool)


def to_b64(vec: np.ndarray) -> str:
    return base64.b64encode(vec.astype(np.float32).tobytes()).decode()


def from_b64(s: str) -> bytes:
    return base64.b64decode(s)


async def wait_for_redis(retries: int = 15, delay: float = 2.0) -> None:
    """Container Redis'e komşu olarak başlasa bile, ağ yolu (bridge/iptables)
    ilk milisaniyelerde henüz oturmamış olabilir. Sabit çökmek yerine birkaç
    kez dene."""
    import asyncio

    import redis.exceptions

    r = client()
    for attempt in range(1, retries + 1):
        try:
            await r.ping()
            return
        except redis.exceptions.ConnectionError as e:
            print(f"[api] redis'e bağlanılamadı ({attempt}/{retries}): {e}")
            await asyncio.sleep(delay)
    raise redis.exceptions.ConnectionError(f"Redis {retries} denemeden sonra hâlâ erişilemez durumda")


async def ensure_schema() -> None:
    """Vektör indeksi ve consumer group'u idempotent şekilde kurar."""
    await wait_for_redis()
    r = client()
    try:
        await r.ft(Keys.CACHE_IDX).create_index(
            fields=[
                TextField("text"),
                TagField("category"),
                TextField("payload"),
                VectorField(
                    "embedding", "HNSW",
                    {"TYPE": "FLOAT32", "DIM": settings.embed_dim,
                     "DISTANCE_METRIC": "COSINE", "M": 16, "EF_CONSTRUCTION": 200},
                ),
            ],
            definition=IndexDefinition(prefix=[Keys.CACHE_PREFIX], index_type=IndexType.HASH),
        )
    except Exception as e:
        if "Index already exists" not in str(e):
            raise

    try:
        await r.xgroup_create(Keys.STREAM, Keys.GROUP, id="0", mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            raise


async def cache_lookup(vec: np.ndarray) -> Optional[tuple[dict, float]]:
    """En yakın komşuyu bul; eşiği geçerse (payload, benzerlik) döndür."""
    r = client()
    q = (
        Query("(*)=>[KNN 1 @embedding $v AS dist]")
        .sort_by("dist")
        .return_fields("payload", "dist", "text")
        .dialect(2)
    )
    try:
        res = await r.ft(Keys.CACHE_IDX).search(q, query_params={"v": vec.astype(np.float32).tobytes()})
    except Exception:
        return None
    if not res.docs:
        return None

    similarity = 1.0 - float(res.docs[0].dist)  # COSINE distance -> benzerlik
    if similarity < settings.cache_threshold:
        return None
    return json.loads(res.docs[0].payload), similarity


async def cache_store(key: str, text: str, vec_bytes: bytes, payload: dict) -> None:
    r = client()
    name = f"{Keys.CACHE_PREFIX}{key}"
    await r.hset(name, mapping={
        "text": text,
        "category": payload.get("category", "diger"),
        "payload": json.dumps(payload, ensure_ascii=False),
        "embedding": vec_bytes,
    })
    await r.expire(name, settings.cache_ttl_seconds)


async def save_ticket(tid: str, data: dict) -> None:
    r = client()
    await r.set(Keys.ticket(tid), json.dumps(data, ensure_ascii=False),
                ex=settings.result_ttl_seconds)
    await r.zadd(Keys.RECENT, {tid: time.time()})
    await r.zremrangebyrank(Keys.RECENT, 0, -201)   # son 200 kayıt
    await r.publish(Keys.EVENTS, json.dumps(data, ensure_ascii=False))


async def get_ticket(tid: str) -> Optional[dict]:
    raw = await client().get(Keys.ticket(tid))
    return json.loads(raw) if raw else None


async def recent_tickets(limit: int = 50) -> list[dict]:
    r = client()
    ids = await r.zrevrange(Keys.RECENT, 0, limit - 1)
    if not ids:
        return []
    rows = await r.mget([Keys.ticket(i.decode()) for i in ids])
    return [json.loads(x) for x in rows if x]


async def bump(field: str) -> None:
    await client().hincrby(Keys.STATS, field, 1)


async def stats() -> dict:
    r = client()
    raw = await r.hgetall(Keys.STATS)
    s = {k.decode(): int(v) for k, v in raw.items()}
    hit, run = s.get("cache_hit", 0), s.get("model_run", 0)
    total = hit + run
    return {
        "cache_hit": hit,
        "model_run": run,
        "hit_rate": round(hit / total, 3) if total else 0.0,
        "queue_depth": await r.xlen(Keys.STREAM),
    }
