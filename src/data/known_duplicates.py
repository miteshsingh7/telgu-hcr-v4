"""Known and verified identical duplicate class pairs in Telugu HCR v4 dataset.

These 34 pairs are Pattern A true duplicates where a bare consonant folder in
hallulu/ contains the exact same base glyph as the 'none' modifier sub-folder in
Guninthamulu/ for that consonant.
"""

from typing import List, Tuple, Dict, Set

CONFIRMED_DUPLICATE_CLASSES: List[Tuple[str, str]] = [
    ("Guninthamulu__kha__ka", "hallulu__ka"),
    ("Guninthamulu__khh__kha", "hallulu__kha"),
    ("Guninthamulu__ga__ga", "hallulu__g"),
    ("Guninthamulu__gha__gha", "hallulu__gh"),
    ("Guninthamulu__ch__ch", "hallulu__ch"),
    ("Guninthamulu__cha__ch", "hallulu__cha"),
    ("Guninthamulu__ja__j", "hallulu__jh"),
    ("Guninthamulu__jh__jh", "hallulu__jha"),
    ("Guninthamulu__ta__ta", "hallulu__ta"),
    ("Guninthamulu__tt__t", "hallulu__th"),
    ("Guninthamulu__d__d", "hallulu__d"),
    ("Guninthamulu__dh__dh", "hallulu__dh"),
    ("Guninthamulu__ana__an", "hallulu__ana"),
    ("Guninthamulu__th__th", "hallulu__tha"),
    ("Guninthamulu__tha__th", "hallulu__thah"),
    ("Guninthamulu__da__da", "hallulu__da"),
    ("Guninthamulu__dha__dh", "hallulu__dha"),
    ("Guninthamulu__na__n", "hallulu__n"),
    ("Guninthamulu__pa__p", "hallulu__P"),
    ("Guninthamulu__pha__p", "hallulu__Ph"),
    ("Guninthamulu__ba__b", "hallulu__b"),
    ("Guninthamulu__bha__bh", "hallulu__bh"),
    ("Guninthamulu__ma__m", "hallulu__m"),
    ("Guninthamulu__ya__y", "hallulu__y"),
    ("Guninthamulu__ra__r", "hallulu__r"),
    ("Guninthamulu__RR__rr", "hallulu__rr"),
    ("Guninthamulu__l__l", "hallulu__l"),
    ("Guninthamulu__ll__l", "hallulu__ll"),
    ("Guninthamulu__va__v", "hallulu__v"),
    ("Guninthamulu__sha__sh", "hallulu__s"),
    ("Guninthamulu__sh__sh", "hallulu__sh"),
    ("Guninthamulu__sa__s", "hallulu__sa"),
    ("Guninthamulu__ha__h", "hallulu__h"),
    ("Guninthamulu__ksh__ksh", "hallulu__ks"),
]

_CANONICAL_LOOKUP: Dict[str, str] = {}
_EQUIVALENCE_GROUPS: Dict[str, Set[str]] = {}

for c1, c2 in CONFIRMED_DUPLICATE_CLASSES:
    canonical = min(c1, c2)
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
