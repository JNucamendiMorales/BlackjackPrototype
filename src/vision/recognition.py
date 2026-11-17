import cv2
import os
import glob


def load_templates(folder):

    templates = {}

    files = glob.glob(os.path.join(folder, "*.png"))

    for file in files:

        name = os.path.splitext(os.path.basename(file))[0]

        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        if image is not None:
            templates[name] = image

    return templates


def recognize_symbol(region_path, templates):

    region = cv2.imread(
        region_path,
        cv2.IMREAD_GRAYSCALE
    )

    if region is None:
        print(f"Error: no se pudo cargar {region_path}")
        return None, 0

    best_name = None
    best_score = -1

    for name, template in templates.items():

        template_height, template_width = template.shape
        region_height, region_width = region.shape

        # El template debe caber dentro de la región.
        if (template_height > region_height or
                template_width > region_width):
            print(f"Template demasiado grande: {name}")
            continue

        # Comparar el template contra la región.
        result = cv2.matchTemplate(
            region,
            template,
            cv2.TM_CCOEFF_NORMED
        )

        _, max_score, _, max_location = cv2.minMaxLoc(result)

        print(f"{name}: {max_score:.3f}")

        if max_score > best_score:
            best_score = max_score
            best_name = name

    return best_name, best_score


# 1. Cargar templates.
value_templates = load_templates("templates/valores")
suit_templates = load_templates("templates/palos")


if not value_templates or not suit_templates:
    print("Error: faltan templates.")
    raise SystemExit


# 2. Reconocer valor y palo.
print("\n========== RECONOCIMIENTO ==========\n")

value, value_score = recognize_symbol(
    "valor_binario.png",
    value_templates
)

print("\n-----------------------------------\n")

suit, suit_score = recognize_symbol(
    "palo_binario.png",
    suit_templates
)


# 3. Mostrar resultado.
print("\n===================================")

if value is not None and suit is not None:

    print(f"Valor reconocido: {value.upper()}")
    print(f"Palo reconocido: {suit.upper()}")

    if value == "as":
        value_display = "AS"
    elif value == "diez":
        value_display = "10"
    else:
        value_display = value.upper()

    print(f"\nCARTA IDENTIFICADA: {value_display} DE {suit.upper()}")

    print(f"\nSimilitud del valor: {value_score:.3f}")
    print(f"Similitud del palo: {suit_score:.3f}")

else:
    print("No se pudo reconocer la carta.")

print("===================================\n")
