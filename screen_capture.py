"""
ScreenCapture
=============
Écoute la touche TAB et capture l'écran à chaque appui.
Fournit le screenshot aux classes suivantes du pipeline.

Utilisation :
    sc = ScreenCapture()
    sc.start()                        # démarre l'écoute en arrière-plan
    sc.on_capture = lambda img: ...   # callback appelé à chaque capture
    sc.stop()                         # arrête l'écoute
"""

import time
import threading
import numpy as np
import keyboard
import mss
import mss.tools
from typing import Callable, Optional


class ScreenCapture:
    """
    Écoute la touche TAB en arrière-plan et capture l'écran entier
    à chaque appui. Appelle on_capture(image) avec un numpy array BGR.

    Le debounce évite les captures multiples si TAB reste enfoncé.
    """

    DEBOUNCE_SECONDS = 0.5   # délai minimum entre deux captures

    def __init__(self):
        self.on_capture: Optional[Callable[[np.ndarray], None]] = None

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_capture_time = 0.0
        self._sct = mss.mss()

    # ------------------------------------------------------------------
    # Démarrage / arrêt
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Démarre l'écoute clavier dans un thread dédié."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()
        print("  [ScreenCapture] Écoute active — appuie sur TAB en jeu.")

    def stop(self) -> None:
        """Arrête l'écoute proprement."""
        self._running = False
        keyboard.unhook_all()
        if self._thread:
            self._thread.join(timeout=2)
        self._sct.close()
        print("  [ScreenCapture] Arrêtée.")

    # ------------------------------------------------------------------
    # Écoute
    # ------------------------------------------------------------------

    def _listen(self) -> None:
        keyboard.on_press_key("tab", self._on_tab_pressed)
        while self._running:
            time.sleep(0.05)

    def _on_tab_pressed(self, event) -> None:
        now = time.time()
        if now - self._last_capture_time < self.DEBOUNCE_SECONDS:
            return
        self._last_capture_time = now

        # Capture dans un thread séparé pour ne pas bloquer l'écoute
        threading.Thread(target=self._capture_and_dispatch, daemon=True).start()

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def _capture_and_dispatch(self) -> None:
        """Prend le screenshot et appelle le callback."""
        image = self._grab()
        print(f"  [ScreenCapture] Screenshot capturé ({image.shape[1]}×{image.shape[0]})")
        if self.on_capture:
            self.on_capture(image)

    def _grab(self) -> np.ndarray:
        """
        Capture l'écran principal avec mss et retourne un array BGR (OpenCV).
        mss est significativement plus rapide que PIL.ImageGrab sur Windows.
        """
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # moniteur principal
            raw = sct.grab(monitor)
            # mss retourne du BGRA — on supprime le canal alpha
            img = np.array(raw)
            return img[:, :, :3]  # BGR

    # ------------------------------------------------------------------
    # Capture manuelle (pour les tests)
    # ------------------------------------------------------------------

    def capture_now(self) -> np.ndarray:
        """Déclenche une capture immédiate sans passer par TAB."""
        image = self._grab()
        if self.on_capture:
            self.on_capture(image)
        return image


# ---------------------------------------------------------------------------
# Test standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import cv2

    print("Test ScreenCapture — appuie sur TAB pour capturer, Ctrl+C pour quitter.\n")

    sc = ScreenCapture()

    def show(img: np.ndarray) -> None:
        cv2.imshow("ScreenCapture — test", img)
        cv2.waitKey(1)

    sc.on_capture = show
    sc.start()

    try:
        while True:
            time.sleep(0.1)
            if cv2.getWindowProperty("ScreenCapture — test", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.waitKey(1)
    except KeyboardInterrupt:
        pass
    finally:
        sc.stop()
        cv2.destroyAllWindows()