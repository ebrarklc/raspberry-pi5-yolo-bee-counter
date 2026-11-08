#!/usr/bin/env python3
"""process_images.py: capture/ klasörünü izler. Gelen her yeni resimdeki arıları sayar,
sonucu SQLite veritabanına yazar. Disk dolmasın diye her KEEP_EVERY resimden birini
capture/done/ klasöründe ÖRNEK olarak saklar, gerisini siler.
Kullanım: python process_images.py      (durdurmak için Ctrl+C)"""
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

WATCH_DIR  = Path("capture")             # capture.sh resimleri buraya bırakır
DONE_DIR   = WATCH_DIR / "done"          # saklanan örnek resimler
MODEL      = "models/bees_640_yolo11n_ncnn_model"
DB_PATH    = "bee_count.db"
IMGSZ, CONF, IOU = 640, 0.3, 0.7         # ölçerek seçtiğimiz ayarlar
KEEP_EVERY = 30                          # her 30 resimden 1'ini sakla (~6 dakikada bir)
ZAMAN = "%Y-%m-%d %H:%M:%S"


def setup_db():
    """Veritabanına bağlan, tablo yoksa oluştur."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bee_counts (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            file         TEXT,
            captured_at  TEXT,
            processed_at TEXT,
            count        INTEGER,
            inference_ms REAL
        )""")
    conn.commit()
    return conn


def capture_time(path):
    """Çekim zamanı: dosya adından (2026-11-08_201617.jpg), olmazsa dosyanın kendi zamanından."""
    try:
        return datetime.strptime(path.stem, "%Y-%m-%d_%H%M%S").strftime(ZAMAN)
    except ValueError:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime(ZAMAN)


def process(path, model, conn):
    frame = cv2.imread(str(path))
    if frame is None:                                  # bozuk ya da yarım dosya
        print(f"⚠️  okunamadı, siliniyor: {path.name}")
        path.unlink()
        return
    if frame.shape[:2] != (IMGSZ, IMGSZ):
        frame = cv2.resize(frame, (IMGSZ, IMGSZ))      # STRETCH: eğitim verisiyle aynı ön işleme

    r = model.predict(frame, imgsz=IMGSZ, conf=CONF, iou=IOU, verbose=False)[0]
    n = len(r.boxes)                                   # sayma = kutu sayısı

    cur = conn.execute(
        "INSERT INTO bee_counts (file, captured_at, processed_at, count, inference_ms) "
        "VALUES (?, ?, ?, ?, ?)",
        (path.name, capture_time(path), datetime.now().strftime(ZAMAN), n, r.speed["inference"]))
    conn.commit()                                      # diske kalıcı yaz

    if cur.lastrowid % KEEP_EVERY == 0:                # her 30. kayıtta resmi sakla
        path.rename(DONE_DIR / path.name)
        durum = "💾 saklandı"
    else:
        path.unlink()                                  # gerisini sil: disk dolmasın
        durum = "🗑️"
    print(f"{datetime.now():%H:%M:%S}  {path.name[:40]:40s}  🐝 {n:3d}  "
          f"({r.speed['inference']:.0f} ms)  {durum}")


def main():
    DONE_DIR.mkdir(parents=True, exist_ok=True)
    conn = setup_db()
    model = YOLO(MODEL, task="detect")
    print(f"İzleniyor: {WATCH_DIR}/  →  {DB_PATH}   (her {KEEP_EVERY} resimden 1'i saklanır, "
          f"durdurmak için Ctrl+C)")
    try:
        while True:
            for path in sorted(WATCH_DIR.glob("*.jpg")):   # sadece .jpg; done/ alt klasörüne bakmaz
                process(path, model, conn)
            time.sleep(1)                                  # saniyede bir kontrol et
    except KeyboardInterrupt:
        print("\nDurduruldu.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
