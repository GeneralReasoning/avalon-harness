"""Agent name generation — `<Color><Philosopher>` (§6.6).

The CEO's name is the reserved literal `"CEO"` and is never minted by
the generator. The literal `"self"` is also reserved (it resolves to the
caller's own agent-trace virtual channel).
"""

from __future__ import annotations

import random
from typing import Iterable

RESERVED_NAMES = {"ceo", "self"}

COLORS = [
    "Red", "Blue", "Green", "Gold", "Silver", "Indigo", "Violet", "Amber", "Crimson",
    "Coral", "Ivory", "Jade", "Onyx", "Pearl", "Ruby", "Sage", "Slate", "Teal", "Topaz",
    "Cyan", "Magenta", "Olive", "Rose", "Saffron", "Maroon", "Azure",
]

PHILOSOPHERS = [
    "Aristotle", "Socrates", "Plato", "Kant", "Hegel", "Hume", "Locke", "Hobbes",
    "Nietzsche", "Spinoza", "Descartes", "Leibniz", "Mill", "Russell", "Wittgenstein",
    "Foucault", "Sartre", "Camus", "Marcus", "Seneca", "Heraclitus", "Parmenides",
    "Epictetus", "Augustine", "Aquinas", "Bacon", "Voltaire",
    "Rousseau", "Kierkegaard", "Schopenhauer", "Heidegger", "Husserl",
    "Derrida", "Deleuze", "Rawls", "Quine", "Frege", "Popper", "Arendt",
    "Beauvoir", "Pythagoras", "Democritus", "Zeno", "Plotinus", "Pascal",
    "Berkeley", "Montaigne", "Machiavelli", "Confucius", "Avicenna",
    "Maimonides", "Ockham", "Anselm", "Comte", "Bergson", "Whitehead", "Marx",
]


def generate_name(used: Iterable[str], *, rng: random.Random | None = None) -> str:
    """Pick a fresh `<Color><Philosopher>` name not in `used` and not reserved."""
    r = rng or random
    used_set = {n.lower() for n in used} | RESERVED_NAMES
    pool = [(c, p) for c in COLORS for p in PHILOSOPHERS]
    r.shuffle(pool)
    for c, p in pool:
        name = f"{c}{p}"
        if name.lower() not in used_set:
            return name
    # Should never reach here at our scale — fall back to a suffix.
    base = pool[0]
    counter = 2
    while True:
        candidate = f"{base[0]}{base[1]}{counter}"
        if candidate.lower() not in used_set:
            return candidate
        counter += 1
