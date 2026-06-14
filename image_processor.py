"""
ImageProcessor
==============
Charge la config des zones (data/zones.json), reçoit un screenshot BGR
de ScreenCapture, découpe chaque slot et transmet les crops au pipeline.

Utilisation :
    ip = ImageProcessor()
    ip.on_crops = lambda crops: ...   # callback appelé après chaque découpe
    sc.on_capture = ip.process        # brancher sur ScreenCapture

    # crops : list[dict]  →  {"zone": Zone, "image": np.ndarray (BGR)}
"""

import json
import os
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Zone (miroir léger de ZoneSelectorTool — pas de dépendance circulaire)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Zone:
    """Slot fixe sur le scoreboard, lu depuis zones.json."""
    team: str    # "ally" | "enemy"
    index: int   # 0-based dans l'équipe
    x: int
    y: int
    width: int
    height: int

    @property
    def label(self) -> str:
        return f"{self.team}#{self.index}"

    @classmethod
    def from_dict(cls, d: dict) -> "Zone":
        return cls(
            team=d["team"],
            index=d["index"],
            x=d["x"],
            y=d["y"],
            width=d["width"],
            height=d["height"],
        )


# ---------------------------------------------------------------------------
# ImageProcessor
# ---------------------------------------------------------------------------

class ImageProcessor:
    """
    Reçoit un screenshot BGR pleine résolution, découpe les slots définis
    dans zones.json et appelle on_crops avec la liste des crops.

    Gestion de résolution :
        Si le screenshot n'est pas à la résolution de référence (zones.json),
        les coordonnées sont mises à l'échelle automatiquement.
    """

    DEFAULT_ZONES_PATH = os.path.join(
        os.path.dirname(__file__), "data", "zones.json"
    )

    def __init__(self, zones_path: Optional[str] = None):
        self.on_crops: Optional[Callable[[list[dict]], None]] = None

        path = zones_path or self.DEFAULT_ZONES_PATH
        self._zones: list[Zone] = []
        self._ref_w: int = 2560
        self._ref_h: int = 1440
        self._load_zones(path)

    # ------------------------------------------------------------------
    # Chargement
    # ------------------------------------------------------------------

    def _load_zones(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[ImageProcessor] zones.json introuvable : {path}\n"
                "Lance d'abord tools/zone_selector_tool.py."
            )
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self._ref_w, self._ref_h = data["resolution"]
        self._zones = [Zone.from_dict(z) for z in data["zones"]]
        print(
            f"  [ImageProcessor] {len(self._zones)} zone(s) chargée(s) "
            f"(référence {self._ref_w}×{self._ref_h})"
        )

    # ------------------------------------------------------------------
    # Mise à l'échelle
    # ------------------------------------------------------------------

    def _scale_factors(self, img: np.ndarray) -> tuple[float, float]:
        """Retourne (sx, sy) pour adapter les coords de référence à l'image."""
        h, w = img.shape[:2]
        return w / self._ref_w, h / self._ref_h

    def _crop_zone(self, img: np.ndarray, zone: Zone, sx: float, sy: float) -> np.ndarray:
        x1 = int(zone.x * sx)
        y1 = int(zone.y * sy)
        x2 = int((zone.x + zone.width) * sx)
        y2 = int((zone.y + zone.height) * sy)

        # Clamp aux dimensions réelles de l'image
        h, w = img.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        return img[y1:y2, x1:x2]

    # ------------------------------------------------------------------
    # Point d'entrée principal
    # ------------------------------------------------------------------

    def process(self, image: np.ndarray) -> list[dict]:
        """
        Découpe le screenshot en crops, appelle on_crops, et retourne la liste.

        Args:
            image: array BGR (H, W, 3) fourni par ScreenCapture.

        Returns:
            list[{"zone": Zone, "image": np.ndarray}]
        """
        sx, sy = self._scale_factors(image)
        crops: list[dict] = []

        for zone in self._zones:
            crop = self._crop_zone(image, zone, sx, sy)
            if crop.size == 0:
                print(f"  [ImageProcessor] ⚠  Crop vide pour {zone.label}, ignoré.")
                continue
            crops.append({"zone": zone, "image": crop})

        print(f"  [ImageProcessor] {len(crops)} crop(s) générés.")

        if self.on_crops:
            self.on_crops(crops)

        return crops


# ---------------------------------------------------------------------------
# Test standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import cv2
    from screen_capture import ScreenCapture

    print("Test ImageProcessor — appuie sur TAB pour capturer, Ctrl+C pour quitter.\n")

    ip = ImageProcessor()
    sc = ScreenCapture()

    def show_crops(crops: list[dict]) -> None:
        for item in crops:
            zone: Zone = item["zone"]
            crop: np.ndarray = item["image"]
            win = f"Crop {zone.label}"
            cv2.imshow(win, crop)
        cv2.waitKey(1)

    ip.on_crops = show_crops
    sc.on_capture = ip.process
    sc.start()

    import time
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        sc.stop()
        cv2.destroyAllWindows()