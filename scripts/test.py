from __future__ import annotations
import argparse
import time 
import requests


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="http://127.0.0.1:8000")
    p.add_argument("--file", default="notes.text")
    args = p.parse_args()

    with open(args.file, "rb") as f:
        r = requests.post(f"{args.api}/docs", files={"file": (args.file, f, "text/plain")})
    print("UPLOAD:", r.status_code, r.text)
    r.raise_status()
    doc_id = r.json()["id"]

    while True:
        s = requests.get(f"{args.api}/docs/{doc_id}/summary")
        print("Summary:", s.status_code, s.text)
        if s.status_code in (200, 409):
            break 
        time.sleep(1)
        
if __name__ == "__main__":
    main()
