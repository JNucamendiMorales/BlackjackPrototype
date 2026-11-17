import cv2
import numpy as np

# 1. Cargar imagen
image = cv2.imread("carta_capturada.jpg")

if image is None:
    print("Error: no se pudo cargar la imagen.")
    exit()

# 2. Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. Reducir ruido
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 4. Detectar bordes
edges = cv2.Canny(blurred, 40, 120)

# 5. Encontrar contornos
contours, _ = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# 6. Buscar posibles cartas
card_contour = None
largest_area = 0

for contour in contours:

    area = cv2.contourArea(contour)

    if area < 5000:
        continue

    perimeter = cv2.arcLength(contour, True)

    approx = cv2.approxPolyDP(
        contour,
        0.02 * perimeter,
        True
    )

    if len(approx) == 4:

        x, y, w, h = cv2.boundingRect(approx)

        aspect_ratio = max(w, h) / min(w, h)

        if 1.35 <= aspect_ratio <= 1.75:

            if area > largest_area:
                largest_area = area
                card_contour = approx

# 7. Comprobar si encontramos una carta
if card_contour is None:

    print("No se pudo detectar una carta.")

else:

    print("Carta detectada correctamente.")

    # Dibujar contorno sobre imagen original
    detected = image.copy()

    cv2.drawContours(
        detected,
        [card_contour],
        -1,
        (0, 255, 0),
        3
    )

    # 8. Obtener las cuatro esquinas
    points = card_contour.reshape(4, 2).astype(np.float32)

    # Ordenar esquinas
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).flatten()

    top_left = points[np.argmin(sums)]
    bottom_right = points[np.argmax(sums)]

    top_right = points[np.argmin(differences)]
    bottom_left = points[np.argmax(differences)]

    ordered_points = np.array([
        top_left,
        top_right,
        bottom_right,
        bottom_left
    ], dtype=np.float32)

    # 9. Definir dimensiones de salida
    width = 570
    height = 890

    destination_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    # 10. Calcular transformación de perspectiva
    matrix = cv2.getPerspectiveTransform(
        ordered_points,
        destination_points
    )

    # 11. Aplicar transformación
    warped = cv2.warpPerspective(
        image,
        matrix,
        (width, height)
    )

    # 12. Guardar resultado
    cv2.imwrite("carta_rectificada.jpg", warped)

    print("Imagen rectificada y guardada.")

    # Mostrar resultados
    cv2.imshow("Carta detectada", detected)
    cv2.imshow("Carta rectificada", warped)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Mostrar bordes para comparar
cv2.imshow("Deteccion de bordes", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
