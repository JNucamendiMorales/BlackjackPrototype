import cv2
import json
import numpy as np
from pathlib import Path

# ==========================================
# CONFIGURACIÓN
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / 'config' / 'game_zones.json'
MIN_PLAYER_AREA = 2500
WINDOW_NAME = 'Blackjack - Hand Index Detection'

if not CONFIG_FILE.exists():
    raise SystemExit('[ERROR] No existe game_zones.json')

with CONFIG_FILE.open('r', encoding='utf-8') as file:
    configuration = json.load(file)

zones = configuration['zones']
camera_config = configuration['camera']
for required_zone in ('PLAYER_1', 'DEALER'):
    if required_zone not in zones:
        raise SystemExit(f'[ERROR] Falta la zona {required_zone}')

def preprocess_region(region):
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def detect_dealer_cards(region):
    result = region.copy()
    binary = preprocess_region(region)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    region_area = region.shape[0] * region.shape[1]
    candidates = []
    for i in range(1, count):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 1200 or area > region_area * 0.45:
            continue
        if w < 35 or h < 55:
            continue
        if not 0.90 <= h / max(1, w) <= 2.60:
            continue
        if area / (w * h) < 0.40:
            continue
        candidates.append((x, y, w, h))

    candidates.sort(key=lambda item: item[0])
    for number, (x, y, w, h) in enumerate(candidates, 1):
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(result, f'CARTA {number}', (x, max(18, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        index_width = max(12, int(w * 0.30))
        index_height = max(18, int(h * 0.32))
        cv2.rectangle(result, (x, y), (x + index_width, y + index_height),
                      (255, 255, 0), 2)
    return result, binary, len(candidates)

def detect_player_indices(region):
    result = region.copy()
    binary = preprocess_region(region)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    candidates = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_PLAYER_AREA:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.012 * perimeter, True)
        points = approx.reshape(-1, 2)
        if len(points) < 4:
            continue

        # La máscara de las cartas es blanca; un vértice exterior
        # convexo tiene un giro con el mismo signo que el contorno.
        orientation = cv2.contourArea(contour, oriented=True)
        if abs(orientation) < 1:
            continue
        contour_sign = np.sign(orientation)

        for i, current in enumerate(points):
            previous = points[i - 1].astype(np.float64)
            current_f = current.astype(np.float64)
            following = points[(i + 1) % len(points)].astype(np.float64)
            incoming = current_f - previous
            outgoing = following - current_f
            len_in = np.linalg.norm(incoming)
            len_out = np.linalg.norm(outgoing)
            if len_in < 12 or len_out < 12:
                continue
            cosine = np.dot(incoming, outgoing) / (len_in * len_out)
            if abs(cosine) > 0.45:
                continue
            cross_z = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
            if np.sign(cross_z) != contour_sign:
                continue  # Esquina cóncava (escalón interior)

            # Una esquina superior izquierda tiene una rama hacia
            # la derecha y otra hacia abajo (con tolerancia a inclinación).
            directions = [previous - current_f, following - current_f]
            right = any(v[0] > 0.65 * np.linalg.norm(v) and
                        abs(v[1]) < 0.65 * np.linalg.norm(v) for v in directions)
            down = any(v[1] > 0.65 * np.linalg.norm(v) and
                       abs(v[0]) < 0.65 * np.linalg.norm(v) for v in directions)
            if not (right and down):
                continue

            x, y = map(int, current)
            if any(np.hypot(x - cx, y - cy) < 18 for cx, cy in candidates):
                continue
            candidates.append((x, y))

    candidates.sort(key=lambda p: (p[1], p[0]))
    for number, (x, y) in enumerate(candidates, 1):
        cv2.circle(result, (x, y), 5, (0, 255, 255), -1)
        cv2.putText(result, f'INDICE {number}', (x + 7, max(14, y - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        # Cuadro aproximado para inspeccionar la esquina visible.
        x2 = min(region.shape[1] - 1, x + 30)
        y2 = min(region.shape[0] - 1, y + 45)
        cv2.rectangle(result, (x, y), (x2, y2), (0, 255, 0), 1)
    return result, binary, len(candidates)

def main():
    camera = cv2.VideoCapture(camera_config['index'])
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config['width'])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config['height'])
    if not camera.isOpened():
        raise SystemExit('[ERROR] No se pudo abrir la cámara.')
    print('\n======================================')
    print(' BLACKJACK - HAND INDEX DETECTION')
    print('======================================')
    print('[INFO] Cámara conectada. Zonas cargadas.')
    print('[INFO] Presiona Q para salir.\n')
    try:
        while True:
            success, frame = camera.read()
            if not success:
                print('[ERROR] No se pudo leer la cámara.')
                break
            frame_height, frame_width = frame.shape[:2]
            for name in ('PLAYER_1', 'DEALER'):
                x1, y1, x2, y2 = zones[name]
                x1, x2 = sorted((max(0, min(int(x1), frame_width)),
                                 max(0, min(int(x2), frame_width))))
                y1, y2 = sorted((max(0, min(int(y1), frame_height)),
                                 max(0, min(int(y2), frame_height))))
                if x2 <= x1 or y2 <= y1:
                    continue
                region = frame[y1:y2, x1:x2]
                if name == 'PLAYER_1':
                    result, binary, count = detect_player_indices(region)
                    label = f'Indices candidatos: {count}'
                else:
                    result, binary, count = detect_dealer_cards(region)
                    label = f'Cartas candidatas: {count}'
                cv2.putText(result, label, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.imshow(f'Blackjack - {name}', result)
                cv2.imshow(f'Binary - {name}', binary)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print('[INFO] Cámara liberada.')


if __name__ == '__main__':
    main()