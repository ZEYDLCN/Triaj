from typing import Literal, Optional
from pydantic import BaseModel, Field

Category = Literal[
    "kargo-teslimat", "iade-degisim", "odeme-fatura",
    "urun-arizasi", "hesap-erisim", "kampanya-indirim", "diger",
]
Priority = Literal["P1", "P2", "P3", "P4"]


class TicketIn(BaseModel):
    text: str = Field(min_length=5, max_length=4000)
    channel: Literal["email", "chat", "form", "telefon"] = "form"
    customer_tier: Literal["standart", "plus", "kurumsal"] = "standart"


class Triage(BaseModel):
    category: Category
    priority: Priority
    sentiment: Literal["olumlu", "notr", "olumsuz", "ofkeli"]
    needs_human: bool
    summary: str
    draft_reply: str


class TicketOut(BaseModel):
    ticket_id: str
    status: Literal["queued", "done", "failed"]
    source: Literal["model", "cache", "pending"] = "pending"
    latency_ms: Optional[int] = None
    similarity: Optional[float] = None
    triage: Optional[Triage] = None
    text: Optional[str] = None
    created_at: Optional[float] = None
