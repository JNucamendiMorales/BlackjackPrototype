"""Exact composition-dependent Blackjack hit/stand EV (single deck, S17).

Unknown dealer hole card is marginalized; no decisions may use its identity.
No double, split, surrender, insurance or natural blackjack bonus.
"""
from collections import Counter
from functools import lru_cache

VALUES = tuple(range(2, 12))

def card_value(card):
    rank = card.split('_')[0].upper()
    return 11 if rank == 'A' else 10 if rank in ('10', 'J', 'Q', 'K') else int(rank)

def hand_state(cards):
    return sum(card_value(c) for c in cards), sum(card_value(c) == 11 for c in cards)

def score(total, aces):
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total

def evaluate(player_hand, dealer_visible, unknown_cards):
    """Compare PEDIR vs PLANTARSE without peeking at dealer hole card.

    unknown_cards must contain all card identities unknown to players,
    including the dealer's actual hole card as a POSSIBILITY, not revealed.
    Never pass the actual hole card or the current deck.used directly.
    Returns EVs in [-1,1], a recommendation, and next-hit bust probability.
    """
    if len(unknown_cards) < 2:
        raise ValueError('Se necesitan al menos dos cartas desconocidas')
    pt, pa = hand_state(player_hand)
    dv = card_value(dealer_visible)
    counts = Counter(card_value(c) for c in unknown_cards)
    initial = tuple(counts.get(v, 0) for v in VALUES)

    @lru_cache(None)
    def dealer_distribution(total, aces, remaining):
        current = score(total, aces)
        if current > 21: return (1.0,) + (0.0,) * 5
        if current >= 17:
            out = [0.0] * 6
            out[current - 16] = 1.0
            return tuple(out)
        n = sum(remaining)
        if not n:
            raise RuntimeError('Baraja agotada antes de terminar turno del dealer')
        out = [0.0] * 6
        for i, c in enumerate(remaining):
            if not c: continue
            rem = remaining[:i] + (c - 1,) + remaining[i+1:]
            dist = dealer_distribution(total + VALUES[i], aces + (VALUES[i] == 11), rem)
            for j in range(6): out[j] += c / n * dist[j]
        return tuple(out)

    @lru_cache(None)
    def stand_ev(total, aces, unknown):
        s = score(total, aces)
        if s > 21: return -1.0
        n = sum(unknown)
        if not n: raise RuntimeError('Falta carta oculta del dealer')
        ev = 0.0
        for i, c in enumerate(unknown):
            if not c: continue
            rem = unknown[:i] + (c - 1,) + unknown[i+1:]
            dist = dealer_distribution(dv + VALUES[i], (dv == 11) + (VALUES[i] == 11), rem)
            reward = dist[0]
            for ds in range(17, 22):
                reward += dist[ds - 16] * ((s > ds) - (s < ds))
            ev += c / n * reward
        return ev

    @lru_cache(None)
    def optimal_ev(total, aces, unknown):
        if score(total, aces) > 21: return -1.0
        stand = stand_ev(total, aces, unknown)
        if sum(unknown) < 2: return stand
        return max(stand, hit_ev(total, aces, unknown))

    @lru_cache(None)
    def hit_ev(total, aces, unknown):
        n = sum(unknown)
        if n < 2: return -1.0
        # Marginal probability of next card is count/n, with hole card
        # excluded physically but unknown. After seeing the drawn card,
        # posterior hole-card possibilities are exactly unknown minus drawn.
        ev = 0.0
        for i, c in enumerate(unknown):
            if not c: continue
            rem = unknown[:i] + (c - 1,) + unknown[i+1:]
            ev += c / n * optimal_ev(total + VALUES[i], aces + (VALUES[i] == 11), rem)
        return ev

    stand = stand_ev(pt, pa, initial)
    hit = hit_ev(pt, pa, initial)
    bust = sum(c / sum(initial) for i, c in enumerate(initial)
               if c and score(pt + VALUES[i], pa + (VALUES[i] == 11)) > 21)
    return {'stand': stand, 'hit': hit,
            'best': 'PEDIR' if hit > stand else 'PLANTARSE',
            'bust_probability': bust}
