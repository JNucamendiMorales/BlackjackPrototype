"""Motor probabilístico de BlackjackPrototype."""

from collections import Counter


RANKS = (
    "2", "3", "4", "5", "6", "7",
    "8", "9", "10", "J", "Q", "K", "A"
)


def card_value(rank):
    """Valor numérico de una carta."""

    if rank == "A":
        return 11

    if rank in ("10", "J", "Q", "K"):
        return 10

    return int(rank)


def probability_of_bust(player_hand, available_cards):
    """
    Calcula la probabilidad de superar 21
    al recibir exactamente una carta.
    """

    if not available_cards:
        return 0.0

    def calculate_score(cards):
        total = 0
        aces = 0

        for card in cards:
            rank = card.split("_")[0]
            total += card_value(rank)

            if rank == "A":
                aces += 1

        while total > 21 and aces > 0:
            total -= 10
            aces -= 1

        return total

    bust_count = 0

    for card in available_cards:
        new_hand = player_hand + [card]

        if calculate_score(new_hand) > 21:
            bust_count += 1

    return bust_count / len(available_cards)

def visible_available_cards(
    full_deck,
    used_cards,
    dealer_hidden=None
):
    """
    Construye las cartas posibles desde la perspectiva
    del jugador.

    used_cards contiene las cartas registradas por NFC.

    Si existe una carta oculta del dealer, se incluye
    nuevamente entre las posibilidades desconocidas.
    """

    known_cards = set(used_cards)

    if dealer_hidden is not None:
        known_cards.discard(dealer_hidden)

    return [
        card for card in full_deck
        if card not in known_cards
    ]
