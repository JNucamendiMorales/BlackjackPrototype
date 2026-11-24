"""
BlackjackPrototype - Seguimiento del dataset PLAYER V6.

No modifica imagenes, modelos ni registros.
"""

import csv
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    ROOT / "dataset" / "training_v6_raw"
    / "player" / "manifest_v6.csv"
)

CARTAS = [
    "2_D",
    "4_C", "4_D",
    "5_C",
    "6_C", "6_T",
    "7_D", "7_T",
    "8_C", "8_T",
    "9_C", "9_D", "9_T", "9_P",
    "10_D",
    "J_C", "J_T", "J_P",
    "Q_C",
    "K_D", "K_T", "K_P",
    "A_P"
]

OBJETIVO = 30


def main():

    if not MANIFEST.exists():
        print("[ERROR] No existe el manifest PLAYER V6.")
        return

    with MANIFEST.open(
        encoding="utf-8", newline=""
    ) as f:
        registros = list(csv.DictReader(f))

    total = 0
    completas = 0

    print("\n========== PROGRESO PLAYER V6 ==========\n")

    print(
        f"{'CARTA':<8}"
        f"{'TOTAL':>7}"
        f"{'40cm':>7}"
        f"{'50cm':>7}"
        f"{'SESIONES':>10}"
        f"{'POSICIONES':>14}"
        f"{'FALTAN':>8}"
    )

    print("-" * 61)

    for carta in CARTAS:

        filas = [
            f for f in registros
            if f["label"] == carta
        ]

        cantidad = len(filas)

        alturas = Counter(
            float(f["height_cm"]) for f in filas
        )

        sesiones = {
            f["session_id"] for f in filas
        }

        posiciones = sorted({
            int(f["index_number"]) for f in filas
        })

        texto_posiciones = (
            ",".join(map(str, posiciones))
            if posiciones else "-"
        )

        faltan = max(0, OBJETIVO - cantidad)

        total += cantidad

        if cantidad >= OBJETIVO:
            completas += 1

        print(
            f"{carta:<8}"
            f"{cantidad:>7}"
            f"{alturas[40.0]:>7}"
            f"{alturas[50.0]:>7}"
            f"{len(sesiones):>10}"
            f"{texto_posiciones:>14}"
            f"{faltan:>8}"
        )

    meta_total = len(CARTAS) * OBJETIVO

    progreso = sum(
        min(
            OBJETIVO,
            sum(f["label"] == carta for f in registros)
        )
        for carta in CARTAS
    )

    clases = {
        f["label"] for f in registros
        if f["label"] in CARTAS
    }

    print("-" * 61)

    print("\nRESUMEN")
    print(f"Cartas disponibles: {len(CARTAS)}")
    print(f"Clases capturadas: {len(clases)}/{len(CARTAS)}")
    print(f"Clases con objetivo alcanzado: {completas}")
    print(f"Imagenes registradas: {total}")
    print(f"Objetivo inicial: {meta_total}")
    print(f"Imagenes faltantes: {meta_total - progreso}")
    print(f"Progreso: {100 * progreso / meta_total:.2f}%")

    print("\n========================================")


if __name__ == "__main__":
    main()
