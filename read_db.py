#!/usr/bin/env python3
"""read_db.py: bee_count.db içeriğini tablo halinde yazdırır ve özet çıkarır."""
import sqlite3

conn = sqlite3.connect("bee_count.db")

print(f"{'id':>3s}  {'çekim':19s}  {'işlenme':19s}  {'arı':>4s}  {'ms':>5s}  dosya")
for row in conn.execute("SELECT id, captured_at, processed_at, count, inference_ms, file "
                        "FROM bee_counts ORDER BY id"):
    i, cap, proc, n, ms, f = row
    print(f"{i:3d}  {cap}  {proc}  {n:4d}  {ms:5.0f}  {f[:30]}")

n_rows, total, avg, mx, avg_ms = conn.execute(
    "SELECT COUNT(*), SUM(count), AVG(count), MAX(count), AVG(inference_ms) FROM bee_counts"
).fetchone()
print(f"\n📊 {n_rows} kayıt | toplam {total} arı | ortalama {avg:.1f} | en fazla {mx} "
      f"| ortalama çıkarım {avg_ms:.0f} ms")
conn.close()
