import cv2

# 1. Cargar la carta rectificada
image = cv2.imread("carta_rectificada.jpg")

if image is None:
    print("Error: no se pudo cargar la carta rectificada.")
    exit()

# 2. Obtener dimensiones de la imagen
height, width = image.shape[:2]

print("Dimensiones de la carta:")
print("Ancho:", width)
print("Alto:", height)

# 3. Definir regiones de interes (ROI)

# Region A: valor de la carta
value_x1 = int(width * 0.02)
value_y1 = int(height * 0.02)
value_x2 = int(width * 0.38)
value_y2 = int(height * 0.30)

# Region B: palo de la carta
suit_x1 = int(width * 0.60)
suit_y1 = int(height * 0.02)
suit_x2 = int(width * 0.99)
suit_y2 = int(height * 0.30)

# 4. Recortar las regiones
value_region = image[
    value_y1:value_y2,
    value_x1:value_x2
]

suit_region = image[
    suit_y1:suit_y2,
    suit_x1:suit_x2
]

# 5. Crear una copia para visualizar los recortes
preview = image.copy()

# Dibujar rectangulo verde alrededor del valor
cv2.rectangle(
    preview,
    (value_x1, value_y1),
    (value_x2, value_y2),
    (0, 255, 0),
    3
)

# Dibujar rectangulo azul alrededor del palo
cv2.rectangle(
    preview,
    (suit_x1, suit_y1),
    (suit_x2, suit_y2),
    (255, 0, 0),
    3
)

# 6. Guardar resultados
cv2.imwrite("region_valor.jpg", value_region)
cv2.imwrite("region_palo.jpg", suit_region)

# 7. Mostrar resultados
cv2.imshow("Carta completa - Regiones ROI", preview)
cv2.imshow("Region A - Valor", value_region)
cv2.imshow("Region B - Palo", suit_region)

print("Regiones extraidas correctamente.")
print("Archivos guardados:")
print("- region_valor.jpg")
print("- region_palo.jpg")

cv2.waitKey(0)
cv2.destroyAllWindows()
