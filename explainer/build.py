#!/usr/bin/env python3
"""Inline data.json (+ oh.json, md.json) into page.src.html -> amylose_joint_explorer.html (the published page)."""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
d = json.loads((H / "data.json").read_text())
for k in ("oh", "md", "res", "pp"):
    if (H / f"{k}.json").exists(): d[k] = json.loads((H / f"{k}.json").read_text())
src = (H / "page.src.html").read_text(); assert src.count("/*DATA*/null") == 1
assert src.count("<!--STRUCT-->") == 1; src = src.replace("<!--STRUCT-->", (H / "struct.svg").read_text())
(H / "amylose_joint_explorer.html").write_text(src.replace("/*DATA*/null", json.dumps(d, separators=(",", ":"))))
print("built", (H / "amylose_joint_explorer.html").stat().st_size, "bytes")
