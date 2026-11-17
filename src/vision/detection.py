import cv2

# Cargar la fotografia
image = cv2.imread("carta_capturada.jpg")

if image is None:
    print("No se pudo cargar la imagen.")
    exit()

# 1. Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Reducir ruido
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 3. Detectar bordes
edges = cv2.Canny(blurred, 50, 150)

# 4. Encontrar contornos
contours, hierarchy = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Copia para dibujar los resultados
result = image.copy()

# 5. Buscar contornos con cuatro esquinas
for contour in contours:

    area = cv2.contourArea(contour)

    if area > 1000:

        perimeter = cv2.arcLength(contour, True)

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if len(approx) == 4:

            cv2.drawContours(
                result,
                [approx],
                -1,
                (0, 255, 0),
                3
            )

            print("Objeto rectangular detectado.")

# Mostrar resultados
cv2.imshow("Original", image)
cv2.imshow("Escala de grises", gray)
cv2.imshow("Deteccion de bordes", edges)
cv2.imshow("Contornos detectados", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
