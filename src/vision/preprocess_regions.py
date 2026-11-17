import cv2

# Cargar las regiones
value = cv2.imread("region_valor.jpg")
suit = cv2.imread("region_palo.jpg")

if value is None or suit is None:
    print("Error: no se encontraron las regiones.")
    exit()

# Convertir a escala de grises
value_gray = cv2.cvtColor(value, cv2.COLOR_BGR2GRAY)
suit_gray = cv2.cvtColor(suit, cv2.COLOR_BGR2GRAY)

# Aplicar desenfoque para reducir ruido
value_blur = cv2.GaussianBlur(value_gray, (3, 3), 0)
suit_blur = cv2.GaussianBlur(suit_gray, (3, 3), 0)

# Aplicar umbralización de Otsu
_, value_binary = cv2.threshold(
    value_blur,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

_, suit_binary = cv2.threshold(
    suit_blur,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# Guardar resultados
cv2.imwrite("valor_binario.png", value_binary)
cv2.imwrite("palo_binario.png", suit_binary)

# Mostrar resultados
cv2.imshow("Valor original", value)
cv2.imshow("Valor binario", value_binary)

cv2.imshow("Palo original", suit)
cv2.imshow("Palo binario", suit_binary)

print("Procesamiento terminado.")

cv2.waitKey(0)
cv2.destroyAllWindows()
