"""
ZoneSelectorTool
================
Outil à exécuter UNE SEULE FOIS pour définir les zones d'icônes de héros
sur le scoreboard Overwatch (TAB) en 2560x1440.

Une zone = un emplacement (slot) sur le scoreboard, pas un héros.
Le héros qui s'y trouve change à chaque partie — c'est HeroRecognizer
qui l'identifie au runtime.

Utilisation :
    python zone_selector_tool.py

Contrôles :
    - Clic gauche + glisser  : dessiner une zone
    - Z                      : annuler la dernière zone
    - S                      : sauvegarder et quitter
    - Échap                  : quitter sans sauvegarder
"""

import cv2
import json
import os
import time
import numpy as np
from PIL import ImageGrab
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------

@dataclass
class Zone:
    """
    Emplacement fixe sur le scoreboard.
    'team'  : à quelle équipe appartient ce slot ("ally" ou "enemy")
    'index' : position dans l'équipe (0 = premier joueur, 1 = deuxième, etc.)
    Les coordonnées sont en pleine résolution (2560x1440).
    """
    team: str    # "ally" | "enemy"
    index: int   # 0-based dans l'équipe
    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Zone":
        return cls(**data)

    @property
    def label(self) -> str:
        return f"{self.team} #{self.index}"


@dataclass
class ZoneConfig:
    """Ensemble des zones définies + métadonnées de résolution."""
    resolution: tuple[int, int]
    zones: list[Zone] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "resolution": list(self.resolution),
            "zones": [z.to_dict() for z in self.zones],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ZoneConfig":
        return cls(
            resolution=tuple(data["resolution"]),
            zones=[Zone.from_dict(z) for z in data.get("zones", [])],
        )

    def next_index(self, team: str) -> int:
        """Retourne le prochain index disponible pour une équipe."""
        existing = [z.index for z in self.zones if z.team == team]
        return max(existing) + 1 if existing else 0


# ---------------------------------------------------------------------------
# ZoneSelectorTool
# ---------------------------------------------------------------------------

class ZoneSelectorTool:
    """
    Interface OpenCV pour sélectionner manuellement les emplacements
    d'icônes de héros sur le scoreboard Overwatch.

    Chaque rectangle dessiné = un slot (ally ou enemy).
    L'index est assigné automatiquement dans l'ordre de dessin.
    """

    WINDOW_NAME = "OW Zone Selector  —  [Glisser] slot  |  [Z] annuler  |  [S] sauvegarder  |  [Échap] quitter"
    OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "zones.json")
    DISPLAY_SCALE = 0.6
    ALLY_COLOR    = (80, 200, 120)   # Vert  BGR
    ENEMY_COLOR   = (80, 100, 220)   # Rouge BGR
    TEXT_COLOR    = (255, 255, 255)
    PREVIEW_COLOR = (200, 200, 50)

    def __init__(self):
        self._screenshot_full: Optional[np.ndarray] = None
        self._screenshot_display: Optional[np.ndarray] = None
        self._config = ZoneConfig(resolution=(2560, 1440))

        # État du dessin en cours
        self._drawing = False
        self._start_x = self._start_y = 0
        self._current_x = self._current_y = 0

        # Équipe active pour le prochain slot (basculée par clic droit)
        self._active_team = "ally"

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def _capture_screen(self) -> np.ndarray:
        pil_img = ImageGrab.grab()
        img = np.array(pil_img)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------
    # Coordonnées : display <-> full resolution
    # ------------------------------------------------------------------

    def _to_full(self, x: int, y: int) -> tuple[int, int]:
        return int(x / self.DISPLAY_SCALE), int(y / self.DISPLAY_SCALE)

    def _to_display(self, x: int, y: int) -> tuple[int, int]:
        return int(x * self.DISPLAY_SCALE), int(y * self.DISPLAY_SCALE)

    # ------------------------------------------------------------------
    # Rendu
    # ------------------------------------------------------------------

    def _render_frame(self) -> np.ndarray:
        frame = self._screenshot_display.copy()

        # Zones déjà validées
        for zone in self._config.zones:
            color = self.ALLY_COLOR if zone.team == "ally" else self.ENEMY_COLOR
            x1, y1 = self._to_display(zone.x, zone.y)
            x2, y2 = self._to_display(zone.x + zone.width, zone.y + zone.height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = zone.label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.TEXT_COLOR, 1, cv2.LINE_AA)

        # Zone en cours de dessin
        if self._drawing:
            color = self.ALLY_COLOR if self._active_team == "ally" else self.ENEMY_COLOR
            cv2.rectangle(frame,
                          (self._start_x, self._start_y),
                          (self._current_x, self._current_y),
                          color, 1)

        # HUD bas de page
        ally_count  = sum(1 for z in self._config.zones if z.team == "ally")
        enemy_count = sum(1 for z in self._config.zones if z.team == "enemy")
        team_color  = self.ALLY_COLOR if self._active_team == "ally" else self.ENEMY_COLOR
        hud = (f"Equipe active : {self._active_team.upper()}  "
               f"[clic droit pour changer]   "
               f"ally:{ally_count}  enemy:{enemy_count}   "
               f"[S] sauvegarder  [Z] annuler")
        cv2.putText(frame, hud, (12, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, team_color, 1, cv2.LINE_AA)

        return frame

    # ------------------------------------------------------------------
    # Callbacks souris
    # ------------------------------------------------------------------

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._start_x, self._start_y = x, y
            self._current_x, self._current_y = x, y

        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            self._current_x, self._current_y = x, y

        elif event == cv2.EVENT_LBUTTONUP and self._drawing:
            self._drawing = False
            self._current_x, self._current_y = x, y
            self._add_zone()

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Clic droit = basculer ally / enemy
            self._active_team = "enemy" if self._active_team == "ally" else "ally"
            print(f"  ↔  Équipe active : {self._active_team.upper()}")

    def _add_zone(self) -> None:
        """Valide le rectangle dessiné et l'ajoute comme nouveau slot."""
        x1, y1 = min(self._start_x, self._current_x), min(self._start_y, self._current_y)
        x2, y2 = max(self._start_x, self._current_x), max(self._start_y, self._current_y)

        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            print("  ⚠  Zone trop petite, ignorée.")
            return

        fx1, fy1 = self._to_full(x1, y1)
        fx2, fy2 = self._to_full(x2, y2)

        index = self._config.next_index(self._active_team)
        zone = Zone(
            team=self._active_team,
            index=index,
            x=fx1, y=fy1,
            width=fx2 - fx1,
            height=fy2 - fy1,
        )
        self._config.zones.append(zone)
        print(f"  ✓  Slot [{zone.label}]  ({fx1}, {fy1})  {zone.width}×{zone.height}px")

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------

    def _save(self) -> None:
        if not self._config.zones:
            print("\n⚠  Aucune zone à sauvegarder.")
            return

        h, w = self._screenshot_full.shape[:2]
        self._config.resolution = (w, h)

        os.makedirs(os.path.dirname(self.OUTPUT_PATH), exist_ok=True)
        with open(self.OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(self._config.to_dict(), f, indent=2, ensure_ascii=False)

        print(f"\n✓  {len(self._config.zones)} slot(s) sauvegardé(s) → {self.OUTPUT_PATH}")
        self._print_summary()

    def _print_summary(self) -> None:
        print("\n── Résumé ─────────────────────────────────────")
        for z in sorted(self._config.zones, key=lambda z: (z.team, z.index)):
            print(f"  [{z.label:<10}]  ({z.x}, {z.y})  {z.width}×{z.height}")
        print("───────────────────────────────────────────────\n")

    # ------------------------------------------------------------------
    # Boucle principale
    # ------------------------------------------------------------------

    def run(self) -> None:
        print("\n══════════════════════════════════════════════")
        print("  OW Zone Selector — 2560×1440")
        print("══════════════════════════════════════════════")
        print("  1. Lance Overwatch, rejoins une partie, appuie sur TAB")
        print("  2. Reviens ici et appuie sur ENTRÉE pour capturer l'écran")
        print()
        print("  Dans la fenêtre :")
        print("   Clic gauche + glisser  → dessiner un slot")
        print("   Clic droit             → basculer ally / enemy")
        print("   Z                      → annuler le dernier slot")
        print("   S                      → sauvegarder et quitter")
        print("   Échap                  → quitter sans sauvegarder")
        input("\n  → Prêt ? Appuyez sur ENTRÉE pour lancer le compte à rebours... ")

        for i in range(5, 0, -1):
            print(f"  Capture dans {i}...", end="\r", flush=True)
            time.sleep(1)
        print("  Capture !                ")

        self._screenshot_full = self._capture_screen()
        h, w = self._screenshot_full.shape[:2]
        print(f"  ✓  Screenshot capturé ({w}×{h})")

        dw = int(w * self.DISPLAY_SCALE)
        dh = int(h * self.DISPLAY_SCALE)
        self._screenshot_display = cv2.resize(self._screenshot_full, (dw, dh))

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, dw, dh)
        cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_callback)

        print("\n  Fenêtre ouverte — commence à dessiner les slots !\n")

        while True:
            frame = self._render_frame()
            cv2.imshow(self.WINDOW_NAME, frame)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("s"):
                self._save()
                break
            elif key == ord("z") and self._config.zones:
                removed = self._config.zones.pop()
                print(f"  ↩  Slot [{removed.label}] annulé.")
            elif key == 27:
                print("\n  Quitter sans sauvegarder.")
                break

        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tool = ZoneSelectorTool()
    tool.run()