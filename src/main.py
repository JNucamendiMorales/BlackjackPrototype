"""BlackjackPrototype: simulación manual y lectura NFC USB ACR122U.
Ejecutar desde la raíz del proyecto: python src/main.py
"""
import time
import sys
from pathlib import Path
from deck_memory import DeckMemory
from expert_bridge import recommend_basic

ROOT = Path(__file__).resolve().parents[1]
MAP_FILE = ROOT / 'config' / 'nfc_cards.json'
RANKS = ('2','3','4','5','6','7','8','9','10','J','Q','K','A')
SUITS = {'C':'Corazones', 'D':'Diamantes', 'T':'Treboles', 'P':'Picas'}


def normalize_card(value):
    """Acepta 7c, jc, 10d y también el formato interno 7_T."""
    value = value.strip().upper().replace(' ', '').replace('-', '_')
    if '_' in value:
        parts = value.split('_')
        if len(parts) != 2:
            raise ValueError('Formato inválido. Ejemplos: 7t, jc, 10d, ap')
        rank, suit = parts
    else:
        rank, suit = value[:-1], value[-1:]
    if rank not in RANKS or suit not in SUITS:
        raise ValueError('Carta inválida. Ejemplos: 7t, jc, 10d, ap')
    return f'{rank}_{suit}'


SUIT_SYMBOLS = {'C': '♥', 'D': '♦', 'T': '♣', 'P': '♠'}


def card_symbol(card):
    rank, suit = card.split('_')
    return f'{rank}{SUIT_SYMBOLS[suit]}'


def card_label(card):
    rank, suit = card.split('_')
    return f'{rank} de {SUITS[suit]}'


def score(hand):
    total = 0
    aces = 0
    for card in hand:
        rank = card.split('_')[0]
        if rank == 'A':
            aces += 1
            total += 11
        elif rank in ('J', 'Q', 'K'):
            total += 10
        else:
            total += int(rank)
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total


def ask_int(prompt, low, high):
    while True:
        try:
            value = int(input(prompt))
            if low <= value <= high:
                return value
        except ValueError:
            pass
        print(f'Escribe un número entre {low} y {high}.')


class ManualReader:
    def read(self):
        while True:
            try:
                return normalize_card(input('Introduce carta (ej. 7t, jc, 3d): '))
            except ValueError as exc:
                print(exc)

    def close(self):
        pass


def parse_nfc_text(text):
    """Convierte el texto NFC al formato interno de Blackjack."""

    symbols = {
        '♥': 'C',
        '♦': 'D',
        '♣': 'T',
        '♠': 'P'
    }

    # Eliminar selectores Unicode de presentación emoji/texto.
    text = text.replace('\ufe0f', '').replace('\ufe0e', '')

    pieces = text.strip().split('|')

    if len(pieces) != 2:
        raise ValueError(f'Texto NFC inesperado: {text!r}')

    rank = pieces[0].strip().upper()
    symbol = pieces[1].strip()

    if symbol not in symbols:
        raise ValueError(f'Palo NFC no reconocido: {symbol!r}')

    return normalize_card(f'{rank}_{symbols[symbol]}')


def ndef_text_from_type2(memory):
    """Extrae el primer registro NDEF de texto de una etiqueta NFC Forum Type 2."""
    # Las primeras cuatro páginas contienen UID y datos del fabricante.
    offset = 16
    while offset < len(memory):
        tag = memory[offset]
        offset += 1
        if tag == 0x00:  # TLV nulo
            continue
        if tag == 0xFE:  # Terminador
            break
        if offset >= len(memory):
            break
        length = memory[offset]
        offset += 1
        if length == 0xFF:
            if offset + 2 > len(memory):
                break
            length = int.from_bytes(memory[offset:offset + 2], 'big')
            offset += 2
        if offset + length > len(memory):
            raise ValueError('TLV NDEF incompleto.')
        value = memory[offset:offset + length]
        offset += length
        if tag != 0x03:
            continue
        # Mensaje NDEF: buscar el primer registro de texto (TNF=Well Known, tipo T).
        pos = 0
        while pos < len(value):
            header = value[pos]
            pos += 1
            if pos + 2 > len(value):
                raise ValueError('Cabecera NDEF incompleta.')
            type_len = value[pos]
            pos += 1
            short_record = bool(header & 0x10)
            if short_record:
                payload_len = value[pos]
                pos += 1
            else:
                if pos + 4 > len(value):
                    raise ValueError('Longitud NDEF incompleta.')
                payload_len = int.from_bytes(value[pos:pos + 4], 'big')
                pos += 4
            id_len = 0
            if header & 0x08:
                id_len = value[pos]
                pos += 1
            if pos + type_len + id_len + payload_len > len(value):
                raise ValueError('Registro NDEF incompleto.')
            record_type = value[pos:pos + type_len]
            pos += type_len + id_len
            payload = value[pos:pos + payload_len]
            pos += payload_len
            if (header & 0x07) == 0x01 and record_type == b'T':
                if not payload:
                    raise ValueError('Registro de texto NDEF vacío.')
                status = payload[0]
                lang_len = status & 0x3F
                if 1 + lang_len > len(payload):
                    raise ValueError('Idioma NDEF inválido.')
                encoding = 'utf-16' if status & 0x80 else 'utf-8'
                return payload[1 + lang_len:].decode(encoding)
        raise ValueError('No se encontró un registro NDEF de texto.')
    raise ValueError('La etiqueta no contiene un mensaje NDEF.')


class NFCReader:
    """Lectura NDEF Type 2 mediante comandos PC/SC del ACR122U."""

    def __init__(self):
        try:
            from smartcard.System import readers
        except ImportError as exc:
            raise RuntimeError('Falta pyscard. Instala PC/SC y luego pyscard.') from exc
        available = readers()
        if not available:
            raise RuntimeError('No se encontró lector PC/SC. Conecta el ACR122U y verifica pcscd.')
        print('\nLectores disponibles:')
        for i, reader in enumerate(available, 1):
            print(f'{i}. {reader}')
        selected = ask_int('Selecciona lector: ', 1, len(available)) - 1
        self.reader = available[selected]
        self.wait_for_removal = False

    def connection(self):
        conn = self.reader.createConnection()
        conn.connect()
        return conn

    @staticmethod
    def transmit(conn, apdu):
        data, sw1, sw2 = conn.transmit(apdu)
        if (sw1, sw2) != (0x90, 0x00):
            raise RuntimeError(f'Lectura PC/SC fallida (SW={sw1:02X}{sw2:02X}).')
        return bytes(data)

    def read_tag(self):
        conn = self.connection()
        uid = self.transmit(conn, [0xFF, 0xCA, 0x00, 0x00, 0x00]).hex().upper()
        # ACR122U: FF B0 00 <pagina> 10 lee 16 bytes (4 páginas Type 2).
        # La capacidad se obtiene del Capability Container (página 3).
        initial = self.transmit(conn, [0xFF, 0xB0, 0x00, 0x00, 0x10])
        if len(initial) != 16 or initial[12] != 0xE1:
            raise ValueError('La etiqueta no parece NFC Forum Type 2 con NDEF.')
        capacity = initial[14] * 8
        if capacity <= 0 or capacity > 4096:
            raise ValueError('Capacidad de etiqueta no válida.')
        total = 16 + capacity
        memory = bytearray(initial)
        for page in range(4, (total + 3) // 4, 4):
            memory.extend(self.transmit(conn, [0xFF, 0xB0, 0x00, page, 0x10]))
        return uid, ndef_text_from_type2(bytes(memory[:total]))

    def tag_present(self):
        try:
            self.connection()
            return True
        except Exception:
            return False

    def read(self):
        print('\nEsperando respuesta...')
        if self.wait_for_removal:
            while self.tag_present():
                time.sleep(0.25)
            self.wait_for_removal = False
        while True:
            try:
                uid, text = self.read_tag()
                card = parse_nfc_text(text)
                self.wait_for_removal = True
                print(f'NFC detectado | UID: {uid} | Texto: {text} | Carta: {card_symbol(card)}')
                return card
            except (ValueError, RuntimeError) as exc:
                # Si hay etiqueta pero el contenido es inválido, informarlo sin asignar carta.
                if self.tag_present():
                    print(f'No se pudo interpretar la etiqueta: {exc}')
                    print('Retira la etiqueta para volver a intentar.')
                    while self.tag_present():
                        time.sleep(0.25)
                time.sleep(0.25)
            except Exception:
                # Sin etiqueta presente, esperar sin inundar la terminal.
                time.sleep(0.25)

    def close(self):
        pass


def refresh_screen():
    """Redibuja el tablero en el mismo lugar (terminal ANSI de VS Code)."""
    sys.stdout.write('\033[2J\033[H')
    sys.stdout.flush()


def dashboard(players, dealer, turn='', status='', reveal_dealer=False, results=None):
    refresh_screen()
    print('═' * 66)
    print('                     BLACKJACK PROTOTYPE')
    print('═' * 66)
    rows = [('DEALER', dealer)] + [(name.upper(), hand) for name, hand in players.items()]
    for name, hand in rows:
        hidden = name == 'DEALER' and not reveal_dealer
        visible = hand[:1] if hidden else hand
        cards = ' | '.join(card_symbol(card) for card in visible)
        if hidden and len(hand) > 1:
            cards += ' | ??'
        if not cards:
            cards = '—'
        total = score(visible) if hidden else score(hand)
        suffix = f'TOTAL: {total}' if hand else 'TOTAL: —'
        if results and name in results:
            suffix += f'  {results[name]}'
        print(f'{name:<12} {cards:<29} {suffix}')
    print('═' * 66)
    print(f'TURNO: {turn or "—"}')
    print(f'ESTADO: {status or "En espera"}')
    print('─' * 66)


def deal(reader, person, deck, players, dealer, reveal_dealer=False):
    while True:

        if deck.is_empty():
            raise RuntimeError(
                "Baraja agotada. Es necesario reiniciar las 52 cartas."
            )

        dashboard(
            players,
            dealer,
            person,
            f'Esperando carta... | Restantes: {deck.remaining()}',
            reveal_dealer
        )

        card = reader.read()

        if not deck.register(card):
            dashboard(
                players,
                dealer,
                person,
                f'{card_symbol(card)} ya fue utilizada en esta baraja.',
                reveal_dealer
            )

            input('Presiona Enter para continuar...')
            continue

        return card


def play(reader, deck, n):
    """Una mano; el dealer es operado físicamente por quien dirige el juego."""
    players = {f'Jugador {i}': [] for i in range(1, n + 1)}
    dealer = []

    for _ in range(2):
        for name, hand in players.items():
            hand.append(deal(reader, name, deck, players, dealer))
            dashboard(players, dealer, name, 'Carta asignada')
        dealer.append(deal(reader, 'Dealer', deck, players, dealer))
        dashboard(players, dealer, 'Dealer', 'Carta asignada')

    for name, hand in players.items():
        if score(hand) == 21:
            dashboard(players, dealer, name, 'BLACKJACK NATURAL - TURNO FINALIZADO')
            input('Presiona ENTER para continuar...')
            continue

        while score(hand) < 21:
            dashboard(
            players,
            dealer,
            name,
            'Esperando respuesta...'
            )

            # SISTEMA EXPERTO EN PROLOG
            # Solo se consulta la carta visible del dealer.
            try:
                recommendation = recommend_basic(
                    hand,
                    dealer[0]
                )

                print('\n' + '═' * 50)
                print('             SISTEMA EXPERTO PROLOG')
                print('═' * 50)
                print(f'Jugador: {name}')
                print(f'Puntuación: {score(hand)}')
                print(f'Estrategia básica: {recommendation}')
                print('═' * 50)

            except Exception as exc:
                print(
                    '\nNo se pudo consultar Prolog: '
                    f'{exc}'
                )

            # El jugador conserva la decisión final.
            choice = input(
                '\n¿Pedir carta (P) o plantarse (S)? '
            ).strip().upper()

            if choice == 'S':
                break

            if choice != 'P':
                continue

            hand.append(
                deal(reader, name, deck, players, dealer)
            )

            dashboard(
                players,
                dealer,
                name,
                'Carta asignada'
            )

        if score(hand) > 21:
            dashboard(players, dealer, name, 'Se pasó de 21 - TURNO FINALIZADO')
            input('Presiona ENTER para continuar...')
        elif score(hand) == 21:
            dashboard(players, dealer, name, '21 PUNTOS - TURNO FINALIZADO')
            input('Presiona ENTER para continuar...')

    dashboard(players, dealer, 'Dealer', 'Carta oculta revelada', True)
    input('Presiona ENTER para continuar...')

    # Se aplica la regla estándar S17 sin mostrar instrucciones al dealer.
    if any(score(hand) <= 21 for hand in players.values()):
        while score(dealer) < 17:
            dealer.append(deal(reader, 'Dealer', deck, players, dealer, True))
            dashboard(players, dealer, 'Dealer', 'Carta asignada', True)

    dashboard(players, dealer, 'Dealer', 'Turno finalizado', True)
    input('Presiona ENTER para mostrar los resultados...')

    dealer_score = score(dealer)
    dealer_blackjack = len(dealer) == 2 and dealer_score == 21
    results = {}
    for name, hand in players.items():
        total = score(hand)
        natural = len(hand) == 2 and total == 21
        if total > 21:
            result = 'PIERDE (se pasó)'
        elif natural and not dealer_blackjack:
            result = 'GANA (blackjack)'
        elif dealer_blackjack and not natural:
            result = 'PIERDE (blackjack dealer)'
        elif total == dealer_score:
            result = 'EMPATE'
        elif dealer_score > 21 or total > dealer_score:
            result = 'GANA'
        else:
            result = 'PIERDE'
        results[name.upper()] = result
    dashboard(players, dealer, 'Fin de partida', 'Resultados finales', True, results)


def end_menu(deck):
    print('\n' + '═' * 42)
    print('             FIN DE PARTIDA')
    print('═' * 42)
    print(f'Cartas utilizadas: {len(deck.used)}')
    print(f'Cartas restantes: {deck.remaining()}')
    print(f'Estado: {deck.status()}')
    print('\n1. Jugar otra partida (misma cantidad de jugadores)')
    print('2. Cambiar número de jugadores')
    print('3. Reiniciar baraja y comenzar nueva partida')
    print('4. Salir')
    return ask_int('Selecciona una opción: ', 1, 4)


def main():
    # Cada ejecución empieza con una baraja nueva, aunque exista un JSON previo.
    deck = DeckMemory()
    deck.reset()
    print('\nNueva sesión: baraja de 52 cartas disponible.')

    print('\n' + '=' * 42)
    print('         BLACKJACK PROTOTYPE')
    print('=' * 42)
    print('1. Simulación manual')
    print('2. Lector NFC USB')
    print('3. Salir')
    choice = ask_int('Opción: ', 1, 3)
    if choice == 3:
        return

    reader = None
    try:
        reader = ManualReader() if choice == 1 else NFCReader()
        n = ask_int('\n¿Cuántos jugadores? (1-3): ', 1, 3)

        while True:
            # Una mano requiere al menos dos cartas por jugador y dos del dealer.
            minimum = 2 * (n + 1)
            if deck.remaining() < minimum:
                print(f'\nNo quedan suficientes cartas para iniciar con {n} jugador(es).')
                print(f'Se necesitan al menos {minimum}; quedan {deck.remaining()}.')
            else:
                if deck.is_low():
                    print(f'\nAVISO: baraja casi agotada ({deck.remaining()} cartas).')
                    print('Se recomienda reiniciar antes de continuar.')
                try:
                    play(reader, deck, n)
                except RuntimeError as exc:
                    # Una mano puede necesitar más cartas que el mínimo inicial.
                    if not deck.is_empty():
                        raise
                    print(f'\n{exc}')
                    print('La mano quedó incompleta; reinicia la baraja para continuar.')

            option = end_menu(deck)
            if option == 1:
                continue
            if option == 2:
                n = ask_int('¿Cuántos jugadores? (1-3): ', 1, 3)
                continue
            if option == 3:
                confirmation = input('¿Confirmas una baraja física nueva/mezclada? (S/N): ').strip().upper()
                if confirmation == 'S':
                    deck.reset()
                    print('Baraja reiniciada: 52 cartas disponibles.')
                else:
                    print('Se conserva la baraja actual.')
                continue
            if option == 4:
                print('Sesión finalizada. La próxima ejecución comenzará con 52 cartas.')
                return
    except KeyboardInterrupt:
        print('\nSesión interrumpida.')
    except Exception as exc:
        print(f'Error: {exc}')
    finally:
        if reader:
            reader.close()
        # La memoria no se arrastra a otra ejecución.
        deck.reset()


if __name__ == '__main__':
    main()
