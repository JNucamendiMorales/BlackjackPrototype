"""Motor de valor esperado de Blackjack con baraja finita."""

from collections import Counter
from functools import lru_cache


def card_rank_value(card):
    rank = card.split("_")[0].upper()

    if rank == "A":
        return 11

    if rank in ("10", "J", "Q", "K"):
        return 10

    return int(rank)


def hand_state(cards):
    """Devuelve puntuación inicial y número de ases."""

    total = 0
    aces = 0

    for card in cards:
        value = card_rank_value(card)
        total += value

        if value == 11:
            aces += 1

    return total, aces


def adjusted_score(total, aces):
    """Ajusta los ases de 11 a 1 cuando es necesario."""

    while total > 21 and aces > 0:
        total -= 10
        aces -= 1

    return total


def expected_value_stand(player_hand, dealer_visible, unknown_cards):
    """
    Calcula el valor esperado de PLANTARSE.

    Considera:
    - Baraja finita.
    - Extracciones sin reemplazo.
    - Carta oculta desconocida.
    - Dealer se planta en todos los 17 (S17).
    """

    player_total, player_aces = hand_state(player_hand)
    player_score = adjusted_score(player_total, player_aces)

    if player_score > 21:
        return -1.0

    counts = Counter(card_rank_value(card) for card in unknown_cards)

    values = tuple(range(2, 12))

    initial_counts = tuple(counts.get(value, 0) for value in values)

    visible_value = card_rank_value(dealer_visible)

    @lru_cache(maxsize=None)
    def dealer_ev(total, aces, remaining):
        """
        EV desde el estado actual del dealer.
        """

        score = adjusted_score(total, aces)

        if score > 21:
            return 1.0

        if score >= 17:
            if player_score > score:
                return 1.0

            if player_score == score:
                return 0.0

            return -1.0

        cards_left = sum(remaining)

        if cards_left == 0:
            if player_score > score:
                return 1.0
            if player_score == score:
                return 0.0
            return -1.0

        result = 0.0

        for index, count in enumerate(remaining):

            if count == 0:
                continue

            value = values[index]

            next_remaining = list(remaining)
            next_remaining[index] -= 1

            next_total = total + value
            next_aces = aces + (1 if value == 11 else 0)

            result += (
                count / cards_left
                * dealer_ev(
                    next_total,
                    next_aces,
                    tuple(next_remaining)
                )
            )

        return result

    cards_left = sum(initial_counts)

    if cards_left == 0:
        raise ValueError("No quedan cartas desconocidas.")

    result = 0.0

    for index, count in enumerate(initial_counts):

        if count == 0:
            continue

        hidden_value = values[index]

        remaining = list(initial_counts)
        remaining[index] -= 1

        total = visible_value + hidden_value

        aces = (
            (1 if visible_value == 11 else 0)
            + (1 if hidden_value == 11 else 0)
        )

        result += (
            count / cards_left
            * dealer_ev(
                total,
                aces,
                tuple(remaining)
            )
        )

    return result