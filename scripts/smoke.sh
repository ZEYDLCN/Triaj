#!/usr/bin/env bash
# Gateway -> API -> Redis -> worker zincirini uçtan uca dener.
set -euo pipefail
BASE=${BASE:-http://localhost:8080}
KEY=${KEY:-triaj-demo-key}

say() { printf "\n\033[1m%s\033[0m\n" "$1"; }

say "1) Sağlık (anahtarsız)"
curl -sf "$BASE/health" && echo

say "2) Anahtarsız istek 401 dönmeli"
curl -s -o /dev/null -w "HTTP %{http_code}\n" "$BASE/v1/stats"

say "3) Talep gönder"
TID=$(curl -sf -X POST "$BASE/v1/tickets" \
  -H "x-api-key: $KEY" -H 'Content-Type: application/json' \
  -d '{"text":"Kartımdan 1249 TL iki kez çekilmiş, acilen iade edin.","channel":"email","customer_tier":"plus"}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["ticket_id"])')
echo "ticket_id=$TID"

say "4) Sonucu bekle"
for i in $(seq 1 60); do
  OUT=$(curl -sf "$BASE/v1/tickets/$TID" -H "x-api-key: $KEY")
  [ "$(echo "$OUT" | python3 -c 'import sys,json;print(json.load(sys.stdin)["status"])')" = "done" ] && break
  sleep 2
done
echo "$OUT" | python3 -m json.tool

say "5) Aynı metni tekrar gönder — cache'ten dönmeli"
curl -sf -X POST "$BASE/v1/tickets" \
  -H "x-api-key: $KEY" -H 'Content-Type: application/json' \
  -d '{"text":"Kartımdan 1249 TL iki kere çekilmiş, lütfen acilen iade edin.","channel":"email","customer_tier":"plus"}' \
  | python3 -m json.tool

say "6) İstatistikler"
curl -sf "$BASE/v1/stats" -H "x-api-key: $KEY" | python3 -m json.tool
