"""Build webapp/dist: a standalone copy of the site for static hosts (Vercel, GitHub Pages, any web server).

index.html holds the page body (styles, markup, scripts); this wraps it in a full HTML document
and copies the model and samples next to it.

    python webapp/build_static.py && cd webapp/dist && npx vercel deploy
"""

import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).parent
DIST = HERE / "dist"

page = (HERE / "index.html").read_text(encoding="utf-8")
head_end = page.index("</style>") + len("</style>")
head, body = page[:head_end], page[head_end:]
head = re.sub(r'<meta charset="utf-8">\s*', "", head)
doc = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
       "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
       f"{head}\n</head>\n<body>\n{body}\n</body>\n</html>\n")

if DIST.exists():
    shutil.rmtree(DIST)
DIST.mkdir()
(DIST / "index.html").write_text(doc, encoding="utf-8")
shutil.copytree(HERE / "samples", DIST / "samples")
shutil.copytree(HERE / "model", DIST / "model", ignore=shutil.ignore_patterns("_ref_*"))
(DIST / "vercel.json").write_text(json.dumps({
    "headers": [
        {"source": "/model/(.*)", "headers": [{"key": "Cache-Control", "value": "public, max-age=3600"}]},
        {"source": "/samples/(.*)", "headers": [{"key": "Cache-Control", "value": "public, max-age=86400"}]},
    ]
}, indent=2))
print("built", DIST, sorted(p.name for p in DIST.iterdir()))
