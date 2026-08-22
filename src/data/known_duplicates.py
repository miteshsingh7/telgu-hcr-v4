"""Known and verified identical duplicate class pairs in Telugu HCR v4 dataset.

These 34 pairs are Pattern A true duplicates where a bare consonant folder in
hallulu/ contains the exact same base glyph as the 'none' modifier sub-folder in
Guninthamulu/ for that consonant.
"""

from typing import List, Tuple, Dict, Set

CONFIRMED_DUPLICATE_CLASSES: List[Tuple[str, str]] = [
    ("Guninthamulu__kha__ka", "hallulu__ka"),       # క (ka)
    ("Guninthamulu__khh__kha", "hallulu__kha"),     # ఖ (kha)
    ("Guninthamulu__ga__ga", "hallulu__g"),         # గ (ga)
    ("Guninthamulu__gha__gha", "hallulu__gh"),      # ఘ (gha)
    ("Guninthamulu__ch__ch", "hallulu__ch"),        # చ (ca)
    ("Guninthamulu__cha__ch", "hallulu__cha"),      # ఛ (cha)
    ("Guninthamulu__ja__j", "hallulu__jh"),         # జ (ja)
    ("Guninthamulu__jh__jh", "hallulu__jha"),       # ఝ (jha)
    ("Guninthamulu__ta__ta", "hallulu__ta"),        # ట (ta retroflex)
    ("Guninthamulu__tt__t", "hallulu__th"),         # ఠ (ttha retroflex)
    ("Guninthamulu__d__d", "hallulu__d"),           # డ (da retroflex)
    ("Guninthamulu__dh__dh", "hallulu__dh"),        # ఢ (ddha retroflex)
    ("Guninthamulu__ana__an", "hallulu__ana"),      # ణ (ana)
    ("Guninthamulu__th__th", "hallulu__tha"),       # త (ta dental)
    ("Guninthamulu__tha__th", "hallulu__thah"),     # థ (tha dental)
    ("Guninthamulu__da__da", "hallulu__da"),        # ద (da dental)
    ("Guninthamulu__dha__dh", "hallulu__dha"),      # ధ (dha dental)
    ("Guninthamulu__na__n", "hallulu__n"),          # న (na)
    ("Guninthamulu__pa__p", "hallulu__P"),          # ప (pa)
    ("Guninthamulu__pha__p", "hallulu__Ph"),        # ఫ (pha)
    ("Guninthamulu__ba__b", "hallulu__b"),          # బ (ba)
    ("Guninthamulu__bha__bh", "hallulu__bh"),       # భ (bha)
    ("Guninthamulu__ma__m", "hallulu__m"),          # మ (ma)
    ("Guninthamulu__ya__y", "hallulu__y"),          # య (ya)
    ("Guninthamulu__ra__r", "hallulu__r"),          # ర (ra)
    ("Guninthamulu__RR__rr", "hallulu__rr"),        # ఱ (rra)
    ("Guninthamulu__l__l", "hallulu__l"),           # ల (la)
    ("Guninthamulu__ll__l", "hallulu__ll"),         # ళ (lla)
    ("Guninthamulu__va__v", "hallulu__v"),          # వ (va)
    ("Guninthamulu__sha__sh", "hallulu__s"),        # శ (sha palatal)
    ("Guninthamulu__sh__sh", "hallulu__sh"),        # ష (ssa retroflex)
    ("Guninthamulu__sa__s", "hallulu__sa"),         # స (sa dental)
    ("Guninthamulu__ha__h", "hallulu__h"),          # హ (ha)
    ("Guninthamulu__ksh__ksh", "hallulu__ks"),      # క్ష (ksha)
]

# Quick lookup sets for canonicalization
_CANONICAL_LOOKUP: Dict[str, str] = {}
_EQUIVALENCE_GROUPS: Dict[str, Set[str]] = {}

for c1, c2 in CONFIRMED_DUPLICATE_CLASSES:
    canonical = min(c1, c2)  # Deterministic alphabetical canonical name
    _CANONICAL_LOOKUP[c1] = canonical
    _CANONICAL_LOOKUP[c2] = canonical
    group = {c1, c2}
    _EQUIVALENCE_GROUPS[c1] = group
    _EQUIVALENCE_GROUPS[c2] = group


def get_canonical_class_name(class_name: str) -> str:
    """Returns canonical name for a class, resolving known duplicate pairs."""
    return _CANONICAL_LOOKUP.get(class_name, class_name)


def get_equivalent_classes(class_name: str) -> Set[str]:
    """Returns the set of all equivalent class names for a given class."""
    return _EQUIVALENCE_GROUPS.get(class_name, {class_name})


def is_known_duplicate_pair(class1: str, class2: str) -> bool:
    """Checks if two class names form a confirmed duplicate pair."""
    return class1 in _EQUIVALENCE_GROUPS and class2 in _EQUIVALENCE_GROUPS[class1]
