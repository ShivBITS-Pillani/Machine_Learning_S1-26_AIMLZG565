"""Download the UCI #468 public dataset into data/ once, before training.

If Virtual Lab internet blocks downloads, manually obtain
'online_shoppers_intention.csv' from UCI and put it in the data/ folder.
"""

import io
import zipfile
from urllib.request import Request, urlopen

from common import DATA_FILE

SOURCE = "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"


def main() -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists():
        print("Dataset already available:", DATA_FILE)
        return
    print("Downloading original dataset from UCI repository...")
    try:
        request = Request(SOURCE, headers={"User-Agent": "Mozilla/5.0 ML-Learning"})
        with urlopen(request, timeout=50) as response:
            archive = response.read()
        with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
            csv_names = [n for n in zipped.namelist() if n.endswith("online_shoppers_intention.csv")]
            if not csv_names:
                raise ValueError("UCI zip did not contain expected CSV")
            DATA_FILE.write_bytes(zipped.read(csv_names[0]))
        print("Saved:", DATA_FILE)
    except Exception as exc:
        raise SystemExit(
            "Automated download failed. Open the official UCI dataset page "
            "in a browser, download online_shoppers_intention.csv, and "
            "save to data/online_shoppers_intention.csv.\n"
            f"Original error: {exc}"
        )


if __name__ == "__main__":
    main()
