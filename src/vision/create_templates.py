import cv2
import os
import sys


def extract_symbol(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: no se pudo abrir {image_path}")
        return None

    # Asegurar que el símbolo sea blanco y el fondo negro.
    _, binary = cv2.threshold(
        image, 127, 255, cv2.THRESH_BINARY
    )

    # Encontrar todos los píxeles blancos.
    points = cv2.findNonZero(binary)

    if points is None:
        print("No se encontró ningún símbolo.")
        return None

    # Obtener el rectángulo que contiene el símbolo completo.
    x, y, w, h = cv2.boundingRect(points)

    symbol = binary[y:y+h, x:x+w]

    # Agregar margen negro.
    symbol = cv2.copyMakeBorder(
        symbol,
        10, 10, 10, 10,
        cv2.BORDER_CONSTANT,
        value=0
    )

    return symbol


if len(sys.argv) != 3:

    print("Uso:")
    print("python create_templates.py valor as")
    print("python create_templates.py palo corazones")
    raise SystemExit


category = sys.argv[1]
name = sys.argv[2]


if category == "valor":
    input_file = "valor_binario.png"
    output_folder = "templates/valores"

elif category == "palo":
    input_file = "palo_binario.png"
    output_folder = "templates/palos"

else:
    print("Categoría incorrecta. Usa 'valor' o 'palo'.")
    raise SystemExit


os.makedirs(output_folder, exist_ok=True)

symbol = extract_symbol(input_file)

if symbol is not None:

    output_file = os.path.join(
        output_folder,
        name + ".png"
    )

    cv2.imwrite(output_file, symbol)

    cv2.imshow("Template creado", symbol)

    print(f"Plantilla guardada: {output_file}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
