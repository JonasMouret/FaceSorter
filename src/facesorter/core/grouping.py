from __future__ import annotations
from pathlib import Path
from typing import List
from PIL import Image, ExifTags
import datetime

def _exif_ts(p: Path) -> float:
    try:
        im = Image.open(p)
        exif = im._getexif()
        if exif:
            tags = {ExifTags.TAGS.get(k): v for k, v in exif.items() if k in ExifTags.TAGS}
            dto = tags.get("DateTimeOriginal") or tags.get("DateTime")
            if dto:
                dto = dto.replace("/", ":")
                dt = datetime.datetime.strptime(dto, "%Y:%m:%d %H:%M:%S")
                return dt.timestamp()
    except Exception:
        pass
    return p.stat().st_mtime

def group_by_time(files: List[Path], window_sec: int) -> List[List[Path]]:
    files_sorted = sorted(files, key=_exif_ts)
    groups, current, last = [], [], None
    for f in files_sorted:
        ts = _exif_ts(f)
        if last is None or abs(ts - last) <= window_sec:
            current.append(f)
        else:
            groups.append(current); current = [f]
        last = ts
    if current:
        groups.append(current)
    return groups
