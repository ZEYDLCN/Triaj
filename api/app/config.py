from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379/0"
    embed_model: str = "intfloat/multilingual-e5-small"
    embed_dim: int = 384

    # Semantik cache: kosinüs benzerliği bu eşiğin üstündeyse modeli hiç çağırma
    cache_threshold: float = 0.93
    cache_ttl_seconds: int = 60 * 60 * 24 * 7

    stream_maxlen: int = 10_000
    result_ttl_seconds: int = 60 * 60 * 24


settings = Settings()


class Keys:
    STREAM = "triaj:stream"           # işlenecek talepler (Redis Stream)
    GROUP = "triaj:workers"           # consumer group
    EVENTS = "triaj:events"           # canlı yayın (pub/sub -> SSE)
    RECENT = "triaj:recent"           # son talepler (sorted set)
    CACHE_IDX = "idx:triaj_cache"     # vektör indeksi
    CACHE_PREFIX = "triaj:cache:"
    STATS = "triaj:stats"             # hash: cache_hit, model_run

    @staticmethod
    def ticket(tid: str) -> str:
        return f"triaj:ticket:{tid}"

    @staticmethod
    def idem(fingerprint: str) -> str:
        return f"triaj:idem:{fingerprint}"
