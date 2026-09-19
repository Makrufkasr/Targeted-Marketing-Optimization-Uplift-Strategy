import os
import urllib.request
import time

DATA_URLS = {
    "portfolio.json": "https://raw.githubusercontent.com/joshuayeung/Starbucks-Capstone-Challenge/master/data/portfolio.json",
    "profile.json": "https://raw.githubusercontent.com/joshuayeung/Starbucks-Capstone-Challenge/master/data/profile.json",
    "transcript.json": "https://raw.githubusercontent.com/joshuayeung/Starbucks-Capstone-Challenge/master/data/transcript.json"
}

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

print(f"Target directory: {RAW_DIR}")

for filename, url in DATA_URLS.items():
    dest_path = os.path.join(RAW_DIR, filename)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"[SKIP] {filename} already exists ({os.path.getsize(dest_path)} bytes)")
        continue
    print(f"[DOWNLOADING] {filename} from {url}...")
    start = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
        f.write(resp.read())
    elapsed = round(time.time() - start, 2)
    print(f"[DONE] {filename} saved ({os.path.getsize(dest_path)} bytes in {elapsed}s)")

print("All datasets downloaded successfully.")
