"""
HeroRecognizer
==============
Charge les templates depuis data/heroes/, compare chaque crop reçu
d'ImageProcessor via template matching OpenCV, et retourne le héros
le plus probable avec son score de confiance.

Utilisation :
    hr = HeroRecognizer()
    hr.on_results = lambda results: ...   # callback appelé après chaque batch
    ip.on_crops = hr.recognize            # brancher sur ImageProcessor

    # results : list[dict]
    #   {
    #     "zone"      : Zone,
    #     "hero"      : str | None,   # None si sous le seuil
    #     "confidence": float,        # 0.0 – 1.0  (TM_CCOEFF_NORMED)
    #   }
"""

import os
import cv2
import numpy as np
from typing import Callable, Optional


class HeroRecognizer:
    """
    Template matching offline contre data/heroes/*.png (ou .jpg).

    Stratégie :
        Pour chaque crop, on redimensionne chaque template à la taille
        du crop et on applique cv2.matchTemplate (TM_CCOEFF_NORMED).
        Le meilleur score global détermine le héros.

    Seuil :
        Si le meilleur score < CONFIDENCE_THRESHOLD, le résultat est
        {"hero": None, "confidence": score} ("unknown").
    """

    DEFAULT_HEROES_PATH = os.path.join(
        os.path.dirname(__file__), "data", "heroes"
    )
    CONFIDENCE_THRESHOLD = 0.75
    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

    def __init__(
        self,
        heroes_path: Optional[str] = None,
        threshold: float = CONFIDENCE_THRESHOLD,
    ):
        self.on_results: Optional[Callable[[list[dict]], None]] = None
        self._threshold = threshold

        path = heroes_path or self.DEFAULT_HEROES_PATH
        self._templates: dict[str, np.ndarray] = {}   # hero_name -> BGR image
        self._load_templates(path)

    # ------------------------------------------------------------------
    # Chargement des templates
    # ------------------------------------------------------------------

    def _load_templates(self, path: str) -> None:
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"[HeroRecognizer] Dossier templates introuvable : {path}\n"
                "Crée data/heroes/ et ajoute une image par héros."
            )

        for filename in sorted(os.listdir(path)):
            stem, ext = os.path.splitext(filename)
            if ext.lower() not in self.SUPPORTED_EXTENSIONS:
                continue
            filepath = os.path.join(path, filename)
            img = cv2.imread(filepath)
            if img is None:
                print(f"  [HeroRecognizer] ⚠  Impossible de lire {filename}, ignoré.")
                continue
            self._templates[stem.lower()] = img

        print(
            f"  [HeroRecognizer] {len(self._templates)} template(s) chargé(s) "
            f"depuis {path}"
        )
        if not self._templates:
            print("  [HeroRecognizer] ⚠  Aucun template — recognize() retournera toujours None.")

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _match_one(self, crop: np.ndarray) -> tuple[Optional[str], float]:
        """
        Compare le crop contre tous les templates.
        Retourne (hero_name, confidence) ou (None, best_score) si sous seuil.
        """
        if not self._templates:
            return None, 0.0

        ch, cw = crop.shape[:2]
        best_hero: Optional[str] = None
        best_score: float = -1.0

        for name, tmpl in self._templates.items():
            # Redimensionner le template à la taille exacte du crop
            resized = cv2.resize(tmpl, (cw, ch), interpolation=cv2.INTER_AREA)

            result = cv2.matchTemplate(crop, resized, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(result)

            if score > best_score:
                best_score = score
                best_hero = name

        if best_score < self._threshold:
            return None, best_score

        return best_hero, best_score

    # ------------------------------------------------------------------
    # Point d'entrée principal
    # ------------------------------------------------------------------

    def recognize(self, crops: list[dict]) -> list[dict]:
        """
        Reçoit la liste de crops d'ImageProcessor, retourne les résultats
        et appelle on_results.

        Args:
            crops: list[{"zone": Zone, "image": np.ndarray}]

        Returns:
            list[{"zone": Zone, "hero": str | None, "confidence": float}]
        """
        results: list[dict] = []

        for item in crops:
            zone = item["zone"]
            crop = item["image"]
            hero, confidence = self._match_one(crop)

            label = hero if hero else "unknown"
            print(
                f"  [HeroRecognizer] {zone.label:<12} → "
                f"{label:<20} (conf: {confidence:.3f})"
            )
            results.append({
                "zone": zone,
                "hero": hero,
                "confidence": confidence,
            })

        if self.on_results:
            self.on_results(results)

        return results


# ---------------------------------------------------------------------------
# Test standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    from screen_capture import ScreenCapture
    from image_processor import ImageProcessor

    print("Test HeroRecognizer — appuie sur TAB pour capturer, Ctrl+C pour quitter.\n")

    hr = HeroRecognizer()
    ip = ImageProcessor()
    sc = ScreenCapture()

    def print_results(results: list[dict]) -> None:
        print("\n── Résultats ───────────────────────────────────")
        for r in results:
            hero = r["hero"] or "unknown"
            print(f"  {r['zone'].label:<12} {hero:<20} {r['confidence']:.3f}")
        print("────────────────────────────────────────────────\n")

    hr.on_results = print_results
    ip.on_crops = hr.recognize
    sc.on_capture = ip.process
    sc.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        sc.stop()