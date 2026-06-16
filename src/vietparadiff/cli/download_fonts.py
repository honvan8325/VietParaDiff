"""Font download helper for VietParaDiff.

The project separates two font roles:

* ``fonts/synthetic/`` contains many Vietnamese-capable fonts used only by
  ``vpd-synthetic`` to render synthetic writer styles.
* ``fonts/gnu/`` contains exactly one GNU/Unicode font used by train/infer to
  render stable grapheme archetypes for the content encoder.

The package does not redistribute font binaries; this command downloads them on
the user's machine.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request

SYNTHETIC_URLS = {
    "NotoSans-Regular.ttf": "https://github.com/notofonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf",
    "NotoSerif-Regular.ttf": "https://github.com/notofonts/noto-fonts/raw/main/hinted/ttf/NotoSerif/NotoSerif-Regular.ttf",
}

# GNU Unifont is a stable Unicode coverage font for archetype rendering.  It is
# intentionally placed in its own folder so the content encoder never depends on
# synthetic style fonts.
GNU_URLS = {
    "unifont-15.1.05.otf": "https://unifoundry.com/pub/unifont/unifont-15.1.05/font-builds/unifont-15.1.05.otf",
}


def _download_many(urls: dict[str, str], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, url in urls.items():
        dest = out / name
        if dest.exists():
            print("exists", dest)
            continue
        print("download", url)
        tmp = out / f".{name}.tmp"
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
        print("saved", dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download project fonts into separate synthetic and GNU archetype folders")
    parser.add_argument("--output", default="fonts", help="Root font directory; creates synthetic/ and gnu/ inside it")
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--gnu-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.output)
    if not args.gnu_only:
        _download_many(SYNTHETIC_URLS, root / "synthetic")
    if not args.synthetic_only:
        _download_many(GNU_URLS, root / "gnu")


if __name__ == "__main__":
    main()
