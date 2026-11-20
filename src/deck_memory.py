import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = ROOT / "data" / "deck_state.json"

RANKS = (
    "2", "3", "4", "5", "6", "7",
    "8", "9", "10", "J", "Q", "K", "A"
)

SUITS = ("C", "D", "T", "P")

FULL_DECK = {
    f"{rank}_{suit}"
    for rank in RANKS
    for suit in SUITS
}


class DeckMemory:

    def __init__(self):
        self.used = set()
        self.load()

    def load(self):
        if not STATE_FILE.exists():
            self.used = set()
            return

        with open(STATE_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        cards = data.get("used_cards", [])

        if not isinstance(cards, list):
            raise ValueError("Estado de baraja inválido.")

        if len(cards) != len(set(cards)):
            raise ValueError("Hay cartas duplicadas en el archivo.")

        if not set(cards).issubset(FULL_DECK):
            raise ValueError("El archivo contiene cartas inválidas.")

        self.used = set(cards)

    def save(self):
        STATE_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        data = {
            "total_cards": 52,
            "used_cards": sorted(self.used),
            "remaining_cards": self.remaining()
        }

        temporary = STATE_FILE.with_suffix(".tmp")

        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False
            )

        temporary.replace(STATE_FILE)

    def register(self, card):
        if card not in FULL_DECK:
            raise ValueError(f"Carta inválida: {card}")

        if card in self.used:
            return False

        self.used.add(card)
        self.save()

        return True

    def remaining(self):
        return 52 - len(self.used)

    def available_cards(self):
        return FULL_DECK - self.used

    def is_low(self):
        return self.remaining() <= 16

    def is_empty(self):
        return self.remaining() == 0

    def reset(self):
        self.used = set()
        self.save()

    def status(self):
        remaining = self.remaining()

        if remaining == 0:
            return "AGOTADA"

        if remaining <= 16:
            return "CASI AGOTADA"

        if remaining <= 20:
            return "REDUCIDA"

        return "NORMAL"
