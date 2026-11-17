import cv2
import csv
import json
import hashlib
import numpy as np

from pathlib import Path
from datetime import datetime


# =====================================================
# CONFIGURACIÓN
# =====================================================

ROOT = Path(__file__).resolve().parents[2]

CONFIG = ROOT / "config" / "game_zones.json"

OUTPUT = ROOT / "dataset" / "indices"

TRAINING_DIR = ROOT / "dataset" / "training"

LABELED_DIR = TRAINING_DIR / "labeled_indices"

LABELS_FILE = TRAINING_DIR / "labels.csv"

OUTPUT.mkdir(parents=True, exist_ok=True)
LABELED_DIR.mkdir(parents=True, exist_ok=True)


with CONFIG.open(encoding="utf-8") as f:
    cfg = json.load(f)

zones = cfg["zones"]
cam_cfg = cfg["camera"]


# =====================================================
# CONFIGURACIÓN DE CARTAS
# =====================================================

VALUES = [
    "A", "2", "3", "4", "5", "6", "7",
    "8", "9", "10", "J", "Q", "K"
]

SUITS = {
    "C": "corazones",
    "D": "diamantes",
    "T": "treboles",
    "P": "picas"
}

WIDTH = 96
HEIGHT = 128


# =====================================================
# VARIABLES
# =====================================================

NAMES = ("PLAYER_1", "DEALER")

selected = {
    name: [] for name in NAMES
}

dragging = {
    name: None for name in NAMES
}

current_frames = {}

snapshot = {}

frozen = False

training_active = False

training_value = None
training_suit = None
training_zone = None

training_count = 0

tracking_box = None
tracking_template = None

last_tracking_score = 0.0

TRACKING_THRESHOLD = 0.72

# Ángulos que probaremos durante el seguimiento
TRACKING_ANGLES = list(range(-45, 46, 5))

# Resultado del seguimiento con rotación
tracking_angle = 0
tracking_quad = None

# =====================================================
# NORMALIZAR IMAGEN
# =====================================================

def normalize_image(image):

    h, w = image.shape[:2]

    scale = min(
        WIDTH / w,
        HEIGHT / h
    )

    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_CUBIC
    )

    canvas = np.full(
        (HEIGHT, WIDTH, 3),
        255,
        dtype=np.uint8
    )

    x = (WIDTH - new_w) // 2
    y = (HEIGHT - new_h) // 2

    canvas[
        y:y + new_h,
        x:x + new_w
    ] = resized

    return canvas


# =====================================================
# GUARDAR ETIQUETA
# =====================================================

def save_label(source, filename, value, suit):

    file_exists = (
        LABELS_FILE.exists()
        and LABELS_FILE.stat().st_size > 0
    )

    with open(
        LABELS_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "source",
                "filename",
                "value",
                "suit"
            ])

        writer.writerow([
            source,
            filename,
            value,
            suit
        ])


# =====================================================
# MOUSE
# =====================================================

def on_mouse(event, x, y, flags, name):

    if event == cv2.EVENT_LBUTTONDOWN:

        dragging[name] = (x, y)

    elif (
        event == cv2.EVENT_LBUTTONUP
        and dragging[name] is not None
    ):

        x0, y0 = dragging[name]

        dragging[name] = None

        image = current_frames.get(name)

        if image is None:
            return

        h, w = image.shape[:2]

        x1, x2 = sorted((
            max(0, min(w, x0)),
            max(0, min(w, x))
        ))

        y1, y2 = sorted((
            max(0, min(h, y0)),
            max(0, min(h, y))
        ))

        if x2 - x1 >= 12 and y2 - y1 >= 18:

            selected[name].append(
                (x1, y1, x2, y2)
            )


# =====================================================
# GUARDADO MANUAL ORIGINAL
# =====================================================

def save_selection(name, image, boxes, batch):

    if image is None:
        return 0

    target = OUTPUT / name.lower()

    target.mkdir(
        parents=True,
        exist_ok=True
    )

    count = 0

    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        filename = target / (
            f"{batch}_{i:02d}.png"
        )

        if cv2.imwrite(str(filename), crop):

            print(f"[OK] {filename}")

            count += 1

    return count


# =====================================================
# INICIAR ENTRENAMIENTO
# =====================================================

def start_training():

    global training_active
    global training_value
    global training_suit
    global training_zone
    global training_count
    global tracking_box
    global tracking_template

    if not frozen:

        print(
            "[AVISO] Congela primero con ESPACIO."
        )

        return

    candidates = [
        name for name in NAMES
        if len(selected[name]) == 1
    ]

    if len(candidates) != 1:

        print(
            "[AVISO] Selecciona exactamente un índice "
            "en una sola zona."
        )

        return

    name = candidates[0]

    print("\n====================================")
    print(" NUEVA SESIÓN DE ENTRENAMIENTO")
    print("====================================")

    value = input(
        "Valor [A,2..10,J,Q,K]: "
    ).strip().upper()

    if value not in VALUES:

        print("[ERROR] Valor incorrecto.")

        return

    suit_code = input(
        "Palo [C,D,T,P]: "
    ).strip().upper()

    if suit_code not in SUITS:

        print("[ERROR] Palo incorrecto.")

        return

    box = selected[name][0]

    image = snapshot.get(name)

    if image is None:

        print("[ERROR] No hay imagen congelada.")

        return

    x1, y1, x2, y2 = box

    template = image[y1:y2, x1:x2].copy()

    if template.size == 0:

        print("[ERROR] Selección vacía.")

        return

    training_value = value
    training_suit = SUITS[suit_code]
    training_zone = name

    training_count = 0

    tracking_box = box
    tracking_template = cv2.cvtColor(
        template,
        cv2.COLOR_BGR2GRAY
    )

    training_active = True

    print("\n[OK] Sesión iniciada.")
    print(
        f"Carta: {training_value} "
        f"de {training_suit}"
    )
    print(f"Zona: {training_zone}")
    print("ESPACIO: reanudar cámara")
    print("G: guardar ejemplo")
    print("F: finalizar sesión")


# =====================================================
# SEGUIMIENTO DEL ÍNDICE
# =====================================================

def rotate_template(template, angle):

    h, w = template.shape[:2]

    center = (w / 2, h / 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    new_w = int(np.ceil(h * sin + w * cos))
    new_h = int(np.ceil(h * cos + w * sin))

    matrix[0, 2] += new_w / 2 - center[0]
    matrix[1, 2] += new_h / 2 - center[1]

    rotated = cv2.warpAffine(
        template,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    mask = np.full(
        (h, w),
        255,
        dtype=np.uint8
    )

    rotated_mask = cv2.warpAffine(
        mask,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return rotated, rotated_mask, matrix

def update_tracking(image):

    global tracking_box
    global tracking_angle
    global tracking_quad
    global last_tracking_score

    tracking_box = None
    tracking_quad = None
    last_tracking_score = 0.0

    if tracking_template is None:
        return None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    best_score = -1.0
    best_result = None

    for angle in TRACKING_ANGLES:

        rotated, mask, matrix = rotate_template(
            tracking_template,
            angle
        )

        th, tw = rotated.shape[:2]

        if gray.shape[0] < th or gray.shape[1] < tw:
            continue

        result = cv2.matchTemplate(
            gray,
            rotated,
            cv2.TM_CCORR_NORMED,
            mask=mask
        )

        # Ignorar resultados numéricamente inválidos
        result = np.nan_to_num(
            result,
            nan=-1.0,
            posinf=-1.0,
            neginf=-1.0
        )

        _, score, _, location = cv2.minMaxLoc(result)

        if score > best_score:

            best_score = score

            best_result = (
                location,
                angle,
                matrix,
                tw,
                th
            )

    last_tracking_score = best_score

    if best_result is None:
        return None

    if best_score < TRACKING_THRESHOLD:
        return None

    location, angle, matrix, tw, th = best_result

    x, y = location

    tracking_angle = angle

    tracking_box = (
        x,
        y,
        x + tw,
        y + th
    )

    original_h, original_w = tracking_template.shape[:2]

    corners = np.array([
        [0, 0],
        [original_w, 0],
        [original_w, original_h],
        [0, original_h]
    ], dtype=np.float32)

    transformed = cv2.transform(
        corners.reshape(1, -1, 2),
        matrix
    )[0]

    transformed[:, 0] += x
    transformed[:, 1] += y

    tracking_quad = transformed

    return tracking_box


# =====================================================
# GUARDAR EJEMPLO DE ENTRENAMIENTO
# =====================================================

def save_training_example():

    global training_count

    if not training_active:

        print("[AVISO] No hay sesión activa.")

        return

    if frozen:

        print(
            "[AVISO] Reanuda la cámara antes de guardar."
        )

        return

    image = current_frames.get(training_zone)

    if image is None:

        print("[ERROR] No hay imagen disponible.")

        return

    box = update_tracking(image)

    if box is None:

        print(
            "[AVISO] No se encontró el índice "
            "con suficiente confianza."
        )

        return

    x1, y1, x2, y2 = box

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return

    normalized = normalize_image(crop)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    unique_id = hashlib.sha256(
        (
            timestamp
            + training_value
            + training_suit
        ).encode("utf-8")
    ).hexdigest()[:12]

    filename = (
        f"{training_value}_"
        f"{training_suit}_"
        f"{unique_id}.png"
    )

    destination = LABELED_DIR / filename

    if not cv2.imwrite(
        str(destination),
        normalized
    ):

        print("[ERROR] No se pudo guardar.")

        return

    source = (
        f"training_capture/"
        f"{training_zone.lower()}/"
        f"{timestamp}.png"
    )

    save_label(
        source,
        filename,
        training_value,
        training_suit
    )

    training_count += 1

    print(
        f"[OK] Ejemplo {training_count} guardado: "
        f"{filename}"
    )


# =====================================================
# FINALIZAR ENTRENAMIENTO
# =====================================================

def finish_training():

    global training_active
    global tracking_template
    global tracking_box

    if not training_active:
        return

    print("\n====================================")
    print(" SESIÓN FINALIZADA")
    print("====================================")

    print(
        f"Carta: {training_value} "
        f"de {training_suit}"
    )

    print(
        f"Ejemplos capturados: {training_count}"
    )

    training_active = False

    tracking_template = None
    tracking_box = None


# =====================================================
# CÁMARA
# =====================================================

camera = cv2.VideoCapture(
    cam_cfg["index"]
)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    cam_cfg["width"]
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    cam_cfg["height"]
)

if not camera.isOpened():

    raise SystemExit(
        "[ERROR] No se pudo abrir la cámara."
    )


# =====================================================
# VENTANAS
# =====================================================

for name in NAMES:

    if name not in zones:
        continue

    window_name = f"Blackjack - {name}"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.setMouseCallback(
        window_name,
        on_mouse,
        name
    )


# =====================================================
# CONTROLES
# =====================================================

print("\n====================================")
print(" BLACKJACK - CAPTURA DE ÍNDICES")
print("====================================")

print("ESPACIO: congelar/reanudar")
print("RATÓN: seleccionar índice")
print("S: guardar recortes manuales")
print("T: iniciar entrenamiento")
print("G: guardar ejemplo de entrenamiento")
print("F: finalizar entrenamiento")
print("C: limpiar selección")
print("Q: salir")


# =====================================================
# BUCLE PRINCIPAL
# =====================================================

try:

    while True:

        if not frozen:

            ok, frame = camera.read()

            if not ok:

                print(
                    "[ERROR] No se pudo leer la cámara."
                )

                break

            current_frames.clear()

            for name in NAMES:

                if name not in zones:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    zones[name]
                )

                h, w = frame.shape[:2]

                x1, x2 = (
                    max(0, min(w, x1)),
                    max(0, min(w, x2))
                )

                y1, y2 = (
                    max(0, min(h, y1)),
                    max(0, min(h, y2))
                )

                if x2 > x1 and y2 > y1:

                    current_frames[name] = frame[
                        y1:y2,
                        x1:x2
                    ].copy()

        # Actualizar seguimiento

        if training_active and not frozen:

            image = current_frames.get(
                training_zone
            )

            if image is not None:

                update_tracking(image)

        # Mostrar ventanas

        for name, img in current_frames.items():

            display = img.copy()

            for i, box in enumerate(
                selected[name],
                1
            ):

                x1, y1, x2, y2 = box

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    display,
                    str(i),
                    (x1, max(16, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

            if (
                training_active
                and name == training_zone
                and tracking_box is not None
                and not frozen
                and last_tracking_score >= TRACKING_THRESHOLD
            ):
                
                points = np.round(
                tracking_quad
                ).astype(np.int32)

                cv2.polylines(
                    display,
                    [points],
                    True,
                    (255, 0, 255),
                    2
                )

            mode = (
                "ENTRENAMIENTO"
                if training_active
                else "MANUAL"
            )

            cv2.putText(
                display,
                (
                    f"{name} | {mode} | "
                    f"{'PAUSA' if frozen else 'LIVE'}"
                ),
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            cv2.imshow(
                f"Blackjack - {name}",
                display
            )

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break

        elif key == 32:

            frozen = not frozen

            if frozen:

                snapshot = {
                    name: img.copy()
                    for name, img
                    in current_frames.items()
                }

                selected = {
                    name: [] for name in NAMES
                }

            else:

                selected = {
                    name: [] for name in NAMES
                }

        elif key == ord("c"):

            selected = {
                name: [] for name in NAMES
            }

        elif key == ord("s"):

            if not frozen:

                print(
                    "[AVISO] Congela primero."
                )

                continue

            batch = datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )

            total = sum(
                save_selection(
                    name,
                    snapshot.get(name),
                    selected[name],
                    batch
                )
                for name in NAMES
            )

            print(
                f"[INFO] {total} recortes guardados."
            )

        elif key == ord("t"):

            start_training()

        elif key == ord("g"):

            save_training_example()

        elif key == ord("f"):

            finish_training()


finally:

    finish_training()

    camera.release()

    cv2.destroyAllWindows()