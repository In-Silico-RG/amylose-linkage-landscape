#!/usr/bin/env python3
"""Inline data.json into page.src.html -> amylose_joint_explorer.html (the published page)."""
from pathlib import Path
H = Path(__file__).resolve().parent
src = (H / "page.src.html").read_text(); assert src.count("/*DATA*/null") == 1
(H / "amylose_joint_explorer.html").write_text(src.replace("/*DATA*/null", (H / "data.json").read_text()))
print("built", (H / "amylose_joint_explorer.html").stat().st_size, "bytes")
