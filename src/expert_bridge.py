"""Conexión entre BlackjackPrototype y SWI-Prolog."""

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RULES_FILE = ROOT / "src" / "expert" / "blackjack_rules.pl"


def prolog_rank(card):
    """Convierte una carta de Python a su valor en Prolog."""
    rank = card.split("_")[0].lower()

    valid_ranks = {
        "2", "3", "4", "5", "6", "7",
        "8", "9", "10", "j", "q", "k", "a"
    }

    if rank not in valid_ranks:
        raise ValueError(f"Carta inválida: {card}")

    return rank


def dealer_value(card):
    """Obtiene el valor visible del dealer."""
    rank = prolog_rank(card)

    if rank == "a":
        return 11

    if rank in ("j", "q", "k"):
        return 10

    return int(rank)


def recommend_basic(player_hand, dealer_visible):
    """Consulta la estrategia básica implementada en Prolog."""

    if not player_hand:
        raise ValueError("El jugador no tiene cartas.")

    if not dealer_visible:
        raise ValueError("Falta la carta visible del dealer.")

    cards = ",".join(prolog_rank(card) for card in player_hand)
    dealer = dealer_value(dealer_visible)

    query = (
        f"estrategia([{cards}], {dealer}, Accion), "
        "write(Accion)"
    )

    result = subprocess.run(
        [
            "swipl",
            "-q",
            "-s",
            str(RULES_FILE),
            "-g",
            query,
            "-t",
            "halt"
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=True
    )

    action = result.stdout.strip()

    if action not in ("pedir", "plantarse"):
        raise RuntimeError(
            f"Respuesta inesperada de Prolog: {action!r}"
        )

    return action.upper()
