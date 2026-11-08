#!/usr/bin/env python3
"""analyze_day.py: bee_count.db (model) ile sim_truth.csv (gerçek) karşılaştırması.
Saatlik özet tablo yazdırır ve günün grafiğini bee_day.png olarak kaydeder."""
import csv
import sqlite3
from datetime import datetime

import matplotlib
matplotlib.use("Agg")                    # ekransız çizim: pencere açmadan dosyaya kaydet
import matplotlib.pyplot as plt
import numpy as np

# ── 1) Veriyi oku ────────────────────────────────────────────────
gercek = {}
with open("sim_truth.csv") as f:
    for row in csv.DictReader(f):
        gercek[row["captured_at"]] = int(row["true_count"])

conn = sqlite3.connect("bee_count.db")
kayitlar = conn.execute(
    "SELECT captured_at, count FROM bee_counts WHERE captured_at LIKE '2026-06-15%' "
    "ORDER BY captured_at").fetchall()

zaman = [datetime.strptime(c, "%Y-%m-%d %H:%M:%S") for c, _ in kayitlar]
model = np.array([n for _, n in kayitlar])
gt = np.array([gercek[c] for c, _ in kayitlar])
hata = model - gt

# ── 2) Saatlik özet: SQL'in GROUP BY gücü ────────────────────────
print("Saat  ölçüm  model ort.")
for saat, adet, ort in conn.execute(
        "SELECT substr(captured_at, 12, 2) AS saat, COUNT(*), AVG(count) "
        "FROM bee_counts WHERE captured_at LIKE '2026-06-15%' GROUP BY saat ORDER BY saat"):
    print(f"{saat}:00  {adet:5d}  {ort:6.1f}  {'█' * int(round(ort))}")
conn.close()

# ── 3) Genel ve dilimlere ayrılmış hata ─────────────────────────
print(f"\nToplam: gerçek {gt.sum()} | model {model.sum()} | fark {model.sum() - gt.sum():+d} "
      f"({100 * (model.sum() - gt.sum()) / gt.sum():+.1f}%)")
print(f"MAE {np.abs(hata).mean():.2f} | bias {hata.mean():+.2f} arı/fotoğraf")
for ad, maske in [("boş (0)", gt == 0), ("sakin (1–5)", (gt >= 1) & (gt <= 5)),
                  ("orta (6–29)", (gt >= 6) & (gt <= 29)), ("kalabalık (≥30)", gt >= 30)]:
    if maske.any():
        print(f"  {ad:16s}: {maske.sum():2d} fotoğraf | bias {hata[maske].mean():+6.2f}")

# ── 4) Grafik ───────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1.3]})
ax1.plot(zaman, gt, color="#E8871E", lw=2.2, label="Gerçek (etiket)")
ax1.plot(zaman, model, color="#1F5FA8", lw=2.2, ls="--", label="Model (YOLO11n · NCNN · Pi 5)")
ax1.fill_between(zaman, gt, model, color="grey", alpha=0.15)
ax1.set_ylabel("Arı sayısı")
ax1.set_title("Simüle edilmiş bir kovan günü · 10 dakikada bir fotoğraf\n"
              "(gerçek kovan fotoğrafları; günün şekli yapay)", fontsize=11)
ax1.legend(loc="upper left")
ax1.grid(alpha=0.3)

renk = ["#C0392B" if h < 0 else "#27AE60" for h in hata]
ax2.bar(zaman, hata, width=0.006, color=renk)
ax2.axhline(0, color="black", lw=0.8)
ax2.set_ylabel("Model − gerçek")
ax2.set_xlabel("Saat")
ax2.grid(alpha=0.3)
ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))

plt.tight_layout()
plt.savefig("bee_day.png", dpi=130)
print("\n📈 Grafik kaydedildi: bee_day.png")
