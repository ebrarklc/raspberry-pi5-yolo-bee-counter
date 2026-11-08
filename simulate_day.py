#!/usr/bin/env python3
"""simulate_day.py: Test kümesindeki GERÇEK kovan fotoğraflarından 'simüle edilmiş bir gün' oluşturur.
06:00–20:00 arası 10 dakikada bir fotoğraf; sabah/akşam sakin, öğlen yoğun bir aktivite eğrisi.
Fotoğraflar capture/ klasörüne çekim zamanını taşıyan adlarla bırakılır; gerçek sayılar sim_truth.csv'ye yazılır.
⚠️ Günün ŞEKLİ yapaydır (resimleri saatlere biz dağıtıyoruz); fotoğraflardaki arılar ve modelin sayımı gerçektir."""
import csv
import math
import shutil
from datetime import datetime, timedelta
from pathlib import Path

IMG_DIR = Path("images/bees_test/test/images")
LBL_DIR = Path("images/bees_test/test/labels")
OUT_DIR = Path("capture")
DAY = datetime(2026, 6, 15)              # simüle edilen gün: bir yaz günü
START_H, END_H, STEP_MIN = 6, 20, 10     # 06:00–20:00, 10 dakikada bir
PEAK_H, WIDTH_H = 13, 3.0                # en yoğun saat ve eğrinin genişliği (saat)


def gercek_sayi(img):
    lbl = LBL_DIR / (img.stem + ".txt")
    return sum(1 for line in open(lbl) if line.strip()) if lbl.exists() else 0


havuz = [(gercek_sayi(p), p) for p in sorted(IMG_DIR.glob("*.jpg"))]
en_fazla = max(n for n, _ in havuz)
kullanilan = set()
OUT_DIR.mkdir(exist_ok=True)

satirlar = []
t = DAY.replace(hour=START_H)
while t < DAY.replace(hour=END_H):
    saat = t.hour + t.minute / 60
    hedef = en_fazla * 0.8 * math.exp(-((saat - PEAK_H) / WIDTH_H) ** 2)   # çan eğrisi

    # hedefe en yakın arı sayısına sahip, daha önce kullanılmamış resmi seç
    n, img = min(((n, p) for n, p in havuz if p not in kullanilan),
                 key=lambda x: abs(x[0] - hedef))
    kullanilan.add(img)

    ad = t.strftime("%Y-%m-%d_%H%M%S")
    gecici = OUT_DIR / f"{ad}.part"
    shutil.copy(img, gecici)
    gecici.rename(OUT_DIR / f"{ad}.jpg")           # atomik: process_images yarım dosya görmez

    satirlar.append((f"{ad}.jpg", t.strftime("%Y-%m-%d %H:%M:%S"), n, img.name))
    t += timedelta(minutes=STEP_MIN)

with open("sim_truth.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "captured_at", "true_count", "source_image"])
    w.writerows(satirlar)

print(f"{len(satirlar)} fotoğraf bırakıldı → {OUT_DIR}/  |  gerçek toplam: {sum(s[2] for s in satirlar)} arı")
print("Sabah 06:00 →", satirlar[0][2], "arı | Öğlen 13:00 →",
      next(s[2] for s in satirlar if s[1].endswith("13:00:00")), "arı | Akşam 19:50 →", satirlar[-1][2], "arı")
