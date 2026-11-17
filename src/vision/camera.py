import cv2
from pathlib import Path


# ==========================================
# CONFIGURACIÓN DEL PROYECTO
# ==========================================

# Obtener automáticamente la raíz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directorio donde guardaremos las fotografías.
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "raw"

# Crear directorio si no existe.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ruta de salida.
OUTPUT_IMAGE = OUTPUT_DIR / "card_capture.jpg"


# ==========================================
# INICIALIZAR CÁMARA
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("[ERROR] No se pudo abrir la webcam.")
    raise SystemExit(1)

print("[INFO] Webcam conectada correctamente.")
print("[INFO] Presiona S para capturar una carta.")
print("[INFO] Presiona Q para salir.")


# ==========================================
# CAPTURA DE VIDEO
# ==========================================

try:

    while True:

        success, frame = camera.read()

        if not success:
            print("[ERROR] No se pudo capturar el fotograma.")
            break

        # Mostrar video.
        cv2.imshow("Blackjack - Camera", frame)

        key = cv2.waitKey(1) & 0xFF

        # Capturar fotografía.
        if key == ord('s'):

            saved = cv2.imwrite(
                str(OUTPUT_IMAGE),
                frame
            )

            if saved:
                print(f"[SUCCESS] Imagen guardada en: {OUTPUT_IMAGE}")
            else:
                print("[ERROR] No se pudo guardar la imagen.")

        # Salir.
        elif key == ord('q'):
            print("[INFO] Cerrando cámara...")
            break

finally:

    camera.release()
    cv2.destroyAllWindows()

    print("[INFO] Cámara liberada correctamente.")