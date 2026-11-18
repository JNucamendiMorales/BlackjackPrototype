"""Medición de estabilidad del detector V2 sin modificar su algoritmo.

Ubicar en src/vision/medir_estabilidad_v2.py y ejecutar desde la raíz:
    python src/vision/medir_estabilidad_v2.py

T: iniciar/reiniciar prueba de 30 segundos | Q: salir.
"""
import time
from pathlib import Path
from datetime import datetime
import csv

import cv2
import detect_dealer_cards as original
import detect_dealer_cards_v2 as v2

DURACION = 30.0
CARTAS_ESPERADAS = 4


def main():
    cfg = original.cam_cfg
    cam = cv2.VideoCapture(cfg['index'])
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, cfg['width'])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg['height'])
    if not cam.isOpened():
        raise SystemExit('No se pudo abrir la cámara. Cierra otros programas que la usen.')

    inicio = None
    registros = []
    finalizado = False
    resumen = ''
    print('MEDICIÓN DE ESTABILIDAD V2 | T: iniciar 30 s | Q: salir')
    print('Coloca cuatro cartas separadas y pulsa T cuando estés listo.')

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                print('Error leyendo la cámara.')
                break
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = map(int, original.zones['DEALER'])
            x1, x2 = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
            y1, y2 = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
            if x2 <= x1 or y2 <= y1:
                raise ValueError('Zona DEALER inválida.')
            dealer = frame[y1:y2, x1:x2].copy()
            _, binaria = original.make_binary(dealer)
            cartas, _ = v2.detectar_reforzado(dealer, binaria)
            cantidad = len(cartas)
            vista = dealer.copy()
            for pts, _ in cartas:
                cv2.polylines(vista, [pts.astype('int32')], True, (0, 255, 0), 2)

            if inicio is not None and not finalizado:
                transcurrido = time.monotonic() - inicio
                if transcurrido <= DURACION:
                    registros.append((transcurrido, cantidad))
                else:
                    finalizado = True
                    total = len(registros)
                    buenos = sum(n == CARTAS_ESPERADAS for _, n in registros)
                    fallos = total - buenos
                    perdidas = 0
                    anterior_bueno = True
                    for _, n in registros:
                        actual_bueno = n == CARTAS_ESPERADAS
                        if anterior_bueno and not actual_bueno:
                            perdidas += 1
                        anterior_bueno = actual_bueno
                    porcentaje = 100 * buenos / total if total else 0
                    resumen = (f'ESTABILIDAD: {porcentaje:.2f}% | '
                               f'{buenos}/{total} fotogramas | '
                               f'{fallos} fallos | {perdidas} interrupciones')
                    print('\n' + resumen)
                    print('Nota: las interrupciones cuentan transiciones de 4 a otro número.')
                    carpeta = original.ROOT / 'debug' / 'estabilidad_v2'
                    carpeta.mkdir(parents=True, exist_ok=True)
                    ruta = carpeta / f'{datetime.now():%Y%m%d_%H%M%S}_estabilidad.csv'
                    with ruta.open('w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['segundo', 'cartas_detectadas', 'cuatro_cartas'])
                        writer.writerows((f'{t:.3f}', n, int(n == CARTAS_ESPERADAS)) for t, n in registros)
                    print('Detalle guardado en:', ruta)

            color = (0, 255, 0) if cantidad == CARTAS_ESPERADAS else (0, 165, 255)
            cv2.putText(vista, f'Cartas: {cantidad}/{CARTAS_ESPERADAS}',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2)
            if inicio is None:
                estado = 'Pulsa T para iniciar'
            elif finalizado:
                estado = f'FINALIZADO - {resumen.split(" | ")[0]} - T repetir'
            else:
                restantes = max(0, DURACION - (time.monotonic() - inicio))
                total = len(registros)
                buenos = sum(n == CARTAS_ESPERADAS for _, n in registros)
                parcial = 100 * buenos / total if total else 0
                estado = f'{restantes:.1f}s | Estabilidad parcial: {parcial:.1f}%'
            cv2.putText(vista, estado, (10, 52), cv2.FONT_HERSHEY_SIMPLEX,
                        .45, (0, 255, 255), 1)
            cv2.imshow('PRUEBA ESTABILIDAD V2 - T iniciar | Q salir', vista)
            tecla = cv2.waitKey(20) & 0xFF
            if tecla == ord('q'):
                break
            if tecla == ord('t'):
                inicio = time.monotonic()
                registros = []
                finalizado = False
                resumen = ''
                print('\nPrueba iniciada: 30 segundos con cuatro cartas.')
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
