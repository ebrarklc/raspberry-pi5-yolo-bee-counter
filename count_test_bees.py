"""count_test_bees.py: Arı modelini (NCNN) Pi'de test kümesinde çalıştırır.
Sayma doğruluğunu (MAE, bias) ve hızı ölçer; iou=0.7 ile iou=0.3'ü karşılaştırır."""
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

MODEL   = "models/bees_640_yolo11n_ncnn_model"
IMG_DIR = Path("images/bees_test/test/images")
LBL_DIR = Path("images/bees_test/test/labels")
IMGSZ, CONF = 640, 0.3            # ← Colab'daki sayma deneyinde seçtiğimiz eşik


def gercek_sayi(img):
    """Cevap anahtarı: etiket dosyasındaki satır sayısı."""
    lbl = LBL_DIR / (img.stem + ".txt")
    return sum(1 for line in open(lbl) if line.strip()) if lbl.exists() else 0


model = YOLO(MODEL, task="detect")
images = sorted(IMG_DIR.glob("*.jpg"))
gt = np.array([gercek_sayi(p) for p in images])
print(f"{len(images)} test resmi | gerçek toplam: {gt.sum()} arı | conf={CONF}\n")

for _ in range(3):                                   # ısınma
    model.predict(images[0], imgsz=IMGSZ, conf=CONF, verbose=False)

for IOU in (0.7, 0.3):                               # Ultralytics varsayılanı vs kitap
    pred, inf, post = [], [], []
    t0 = time.perf_counter()
    for img in images:
        r = model.predict(img, imgsz=IMGSZ, conf=CONF, iou=IOU, verbose=False)[0]
        pred.append(len(r.boxes))                    # SAYMA = kutu sayısı
        inf.append(r.speed["inference"])
        post.append(r.speed["postprocess"])
    sure = time.perf_counter() - t0
    pred = np.array(pred)
    hata = pred - gt

    kal, sak, bos = gt >= 30, (gt >= 1) & (gt <= 5), gt == 0
    print(f"── iou = {IOU} ──────────────────────────────")
    print(f"  tahmin toplam : {pred.sum()} arı  ({100 * (pred.sum() - gt.sum()) / gt.sum():+.1f}%)")
    print(f"  MAE           : {np.abs(hata).mean():.2f} arı/resim")
    print(f"  bias          : {hata.mean():+.2f} arı/resim")
    print(f"  kalabalık (≥30 arı, {kal.sum():3d} resim): bias {hata[kal].mean():+6.2f}")
    print(f"  sakin   (1–5 arı, {sak.sum():3d} resim): bias {hata[sak].mean():+6.2f}")
    print(f"  boş     (0 arı,   {bos.sum():3d} resim): resim başına hayalet arı {pred[bos].mean():.2f}")
    print(f"  hız           : inference {np.mean(inf):.1f} ms | postprocess {np.mean(post):.1f} ms"
          f" | toplam {sure:.0f} s ({len(images) / sure:.1f} resim/s)\n")
