from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

@dataclass(frozen=True)
class SorterConfig:
    people_dir: Path
    input_dir: Path
    output_dir: Path
    unknown_dirname: str = "_Unknown"
    noface_dirname: str = "_NoFace"
    min_face_size: int = 24
    topk: int = 5
    embed_norm: bool = True
    match_threshold: float = 0.45
    ambiguous_margin: float = 0.05
    group_window_sec: int = 4
    upscale_factors: Tuple[float, float] = (1.5, 2.0)
    duplicate_multi_faces: bool = False
    ctx_id: int = -1         # -1 CPU, 0 GPU#0
    poll_seconds: int = 5
    move_instead_copy: bool = True

    @staticmethod
    def from_env(
        people: str | None = None,
        input_: str | None = None,
        output: str | None = None,
        **overrides,
    ) -> "SorterConfig":
        import os
        base = Path.cwd()
        cfg = SorterConfig(
            people_dir=Path(people or os.getenv("FS_PEOPLE", base/"people")),
            input_dir=Path(input_ or os.getenv("FS_INPUT", base/"input_photos")),
            output_dir=Path(output or os.getenv("FS_OUTPUT", base/"output_photos")),
        )
        return dataclass_replace(cfg, **overrides)

def dataclass_replace(cfg: SorterConfig, **changes) -> SorterConfig:
    # petite util pour garder l'immutabilité
    return type(cfg)(**{**cfg.__dict__, **changes})
