from pathlib import Path

p = Path("data/processed/scaler_y.pkl")

print("경로:", p.resolve())
print("존재:", p.exists())

if p.exists():
    print("크기:", p.stat().st_size)
    print("첫 100바이트:", repr(p.read_bytes()[:100]))