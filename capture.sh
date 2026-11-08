#!/bin/bash
# capture.sh: Her INTERVAL saniyede bir, TAM görüş alanında 16:9 fotoğraf çeker.
# Fotoğraflar önce .part olarak yazılır, bitince .jpg yapılır (process_images.py yarım dosya görmesin).

OUT_DIR="$HOME/Documents/YOLO/capture"
INTERVAL=10

mkdir -p "$OUT_DIR"
echo "📷 Çekim başladı: her ${INTERVAL} sn → $OUT_DIR   (durdurmak için Ctrl+C)"

while true; do
    DATE=$(date +"%Y-%m-%d_%H%M%S")
    TMP="$OUT_DIR/$DATE.part"

    if rpicam-jpeg -n -t 1500 \
           --mode 1640:1232:10:P --viewfinder-mode 1640:1232:10:P \
           --width 1280 --height 720 \
           -o "$TMP" > /dev/null 2>&1; then
        mv "$TMP" "$OUT_DIR/$DATE.jpg"
        echo "$(date +%H:%M:%S)  📷 $DATE.jpg"
    else
        echo "$(date +%H:%M:%S)  ⚠️  çekim başarısız (kamera bağlı mı?)"
        rm -f "$TMP"
    fi

    sleep "$INTERVAL"
done
