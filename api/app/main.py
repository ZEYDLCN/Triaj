import asyncio
import hashlib
import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from . import embedder, store
from .config import Keys, settings
from .schemas import TicketIn, TicketOut

REQS = Counter("triaj_requests_total", "Alınan talep sayısı", ["source"])
LAT = Histogram("triaj_intake_seconds", "Talep kabul süresi")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await store.ensure_schema()
    await asyncio.to_thread(embedder.load)   # modeli önden ısıt
    yield


app = FastAPI(title="Triaj API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    await store.client().ping()
    return {"status": "ok", "model": settings.embed_model}


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/stats")
async def get_stats():
    return await store.stats()


@app.post("/v1/tickets", response_model=TicketOut, status_code=202)
@LAT.time()
async def create_ticket(payload: TicketIn, request: Request):
    tid = uuid.uuid4().hex[:12]
    now = time.time()
    t0 = time.perf_counter()

    vec = await embedder.encode(payload.text)

    # 1) Semantik cache: aynı derdi anlatan eski bir talep var mı?
    hit = await store.cache_lookup(vec)
    if hit:
        triage, similarity = hit
        record = {
            "ticket_id": tid, "status": "done", "source": "cache",
            "similarity": round(similarity, 4),
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "triage": triage, "text": payload.text, "channel": payload.channel,
            "customer_tier": payload.customer_tier, "created_at": now,
        }
        await store.save_ticket(tid, record)
        await store.bump("cache_hit")
        REQS.labels(source="cache").inc()
        return record

    # 2) Cache ıskaladı -> kuyruğa bırak, worker modeli çalıştırsın
    record = {
        "ticket_id": tid, "status": "queued", "source": "pending",
        "text": payload.text, "channel": payload.channel,
        "customer_tier": payload.customer_tier, "created_at": now,
    }
    await store.save_ticket(tid, record)
    await store.client().xadd(
        Keys.STREAM,
        {
            "ticket_id": tid,
            "text": payload.text,
            "channel": payload.channel,
            "customer_tier": payload.customer_tier,
            "embedding": store.to_b64(vec),           # worker tekrar embed etmesin
            "cache_key": hashlib.sha1(payload.text.encode()).hexdigest()[:16],
            "enqueued_at": str(now),
        },
        maxlen=settings.stream_maxlen, approximate=True,
    )
    REQS.labels(source="model").inc()
    return record


@app.get("/v1/tickets/{ticket_id}", response_model=TicketOut)
async def read_ticket(ticket_id: str):
    data = await store.get_ticket(ticket_id)
    if not data:
        raise HTTPException(404, "Talep bulunamadı veya süresi doldu.")
    return data


@app.get("/v1/tickets")
async def list_tickets(limit: int = 50):
    return {"items": await store.recent_tickets(min(limit, 200))}


@app.get("/events")
async def events(request: Request):
    """Sonuçlar hazır oldukça panele düşen SSE akışı."""
    async def gen():
        pubsub = store.client().pubsub()
        await pubsub.subscribe(Keys.EVENTS)
        try:
            yield b": baglanti-kuruldu\n\n"
            while True:
                if await request.is_disconnected():
                    break
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=15.0)
                if msg is None:
                    yield b": ping\n\n"          # proxy timeout'una karşı
                    continue
                yield b"data: " + msg["data"] + b"\n\n"
        finally:
            await pubsub.unsubscribe(Keys.EVENTS)
            await pubsub.aclose()

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
