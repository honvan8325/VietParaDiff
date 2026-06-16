"""Font download helper.

This script intentionally performs downloads only when the user runs it on their
machine.  The source package itself does not redistribute font binaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request

URLS = {
    "NotoSans-Regular.ttf": "https://github.com/notofonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf",
    "NotoSerif-Regular.ttf": "https://github.com/notofonts/noto-fonts/raw/main/hinted/ttf/NotoSerif/NotoSerif-Regular.ttf",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download open Vietnamese-capable fonts into fonts/")
    parser.add_argument("--output", default="fonts")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        dest = out / name
        if dest.exists():
            print("exists", dest)
            continue
        print("download", url)
        urllib.request.urlretrieve(url, dest)
        print("saved", dest)


if __name__ == "__main__":
    main()
