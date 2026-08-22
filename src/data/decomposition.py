"""Telugu Grapheme Decomposition, Dynamic Class Derivation, and Recombination.

Decomposes compound Telugu aksharas into three structural primitives:
  1. Base Akshara (Vowel or Consonant)
  2. Vowel Modifier (Gunintham matra or 'none')
  3. Subscript Conjunct (Vattu / Othu or 'none')

Provides:
  - Exact alias resolution with case-insensitive fallbacks
  - Zero-error dataset validation and dynamic vocabulary derivation
  - Reverse recombination index and Constrained Maximum-Likelihood Decoding
"""

import json
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional
import numpy as np

# Initial Reference Primitives (Used for consistent ordering)
CANONICAL_BASE_LETTERS: List[str] = [
    # Achulu (Vowels) - 16
    "అ", "ఆ", "ఇ", "ఈ", "ఉ", "ఊ", "ఋ", "ౠ",
    "ఎ", "ఏ", "ఐ", "ఒ", "ఓ", "ఔ", "అం", "అః",
    # Hallulu (Consonants) - 36
    "క", "ఖ", "గ", "ఘ", "ఙ",
    "చ", "ఛ", "జ", "ఝ", "ఞ",
    "ట", "ఠ", "డ", "ఢ", "ణ",
    "త", "థ", "ద", "ధ", "న",
    "ప", "ఫ", "బ", "భ", "మ",
    "య", "ర", "ఱ", "ల", "ళ", "వ",
    "శ", "ష", "స", "హ", "క్ష"
]

CANONICAL_VOWEL_MODIFIERS: List[str] = [
    "none",   # తలకట్టు / అ (క)
    "aa",     # దీర్ఘం ా (కా)
    "i",      # గుడి ి (కి)
    "ii",     # గుడిదీర్ఘం ీ (కీ)
    "u",      # కొమ్ము ు (కు)
    "uu",     # కొమ్ముదీర్ఘం ూ (కూ)
    "ru",     # వట్రుసుడి ృ (కృ)
    "ruu",    # వట్రుసుడి దీర్ఘం ౄ (కౄ)
    "e",      # ఎత్వం ె (కె)
    "ee",     # ఏత్వం ే (కే)
    "ai",     # ఐత్వం ై (కై)
    "o",      # ఒత్వం ొ (కొ)
    "oo",     # ఓత్వం ో (కో)
    "au",     # ఔత్వం ౌ (కౌ)
    "am",     # సున్నా ం (కం)
    "ah",     # విసర్గ ః (కః)
]

CANONICAL_CONJUNCT_MODIFIERS: List[str] = [
    "none",   # Standard character (no subscript)
    "k", "kh", "g", "gh", "gna",
    "c", "ch", "j", "jh", "jna",
    "t", "th", "d", "dh", "ana",
    "th_dental", "d_dental", "dh_dental", "n",
    "p", "ph", "b", "bh", "m",
    "y", "r", "l", "v", "sh", "sha", "s", "h", "ll", "ks", "rr", "z"
]

# Alias dictionaries with exhaustive coverage for dataset quirks
CONSONANT_ALIASES = {
    "kha": "క", "khh": "ఖ", "kh": "ఖ", "ka": "క", "k": "క",
    "ga": "గ", "g": "గ", "gha": "ఘ", "gh": "ఘ", "gna": "ఙ",
    "cha": "చ", "ch": "ఛ", "chh": "ఛ", "c": "చ",
    "ja": "జ", "j": "జ", "jha": "ఝ", "jh": "ఝ", "jna": "ఞ",
    "ta": "ట", "t": "ట", "tt": "ట", "tha": "ఠ", "th": "ఠ", "thah": "ఠ",
    "da": "డ", "d": "డ", "dha": "ఢ", "dh": "ఢ", "ana": "ణ", "an": "ణ", "nn": "ణ",
    "th_dental": "త", "tha_dental": "త", "d_dental": "ద", "da_dental": "ద",
    "na": "న", "n": "న",
    "pa": "ప", "p": "ప", "P": "ప", "pha": "ఫ", "ph": "ఫ", "Ph": "ఫ",
    "ba": "బ", "b": "బ", "bha": "భ", "bh": "భ", "ma": "మ", "m": "మ",
    "ya": "య", "y": "య",
    "ra": "ర", "r": "ర", "rr": "ఱ", "RR": "ఱ",
    "la": "ల", "l": "ల", "ll": "ళ",
    "va": "వ", "v": "వ",
    "sha": "ష", "sh": "శ",
    "sa": "స", "s": "స",
    "ha": "హ", "h": "హ",
    "ksh": "క్ష", "ks": "క్ష", "z": "క"
}

VOWEL_ALIASES = {
    "a": "అ", "aa": "ఆ", "i": "ఇ", "ii": "ఈ", "u": "ఉ", "uu": "ఊ",
    "ru": "ఋ", "ruu": "ౠ", "e": "ఎ", "ee": "ఏ", "ai": "ఐ",
    "o": "ఒ", "oo": "ఓ", "au": "ఔ", "ao": "ఔ", "am": "అం", "ah": "అః"
}

MOD_ALIASES = {
    "a": "none", "aa": "aa", "i": "i", "ii": "ii", "u": "u", "uu": "uu",
    "ru": "ru", "ruu": "ruu", "e": "e", "ee": "ee", "ai": "ai",
    "o": "o", "oo": "oo", "au": "au", "ao": "au", "ow": "au", "am": "am", "ah": "ah",
    "m": "am", "r": "ru", "rrr": "ruu", "R": "ru", "RRA": "aa", "RRI": "i", "RRII": "ii",
    "RRU": "u", "RRUU": "uu", "rre": "e", "rree": "ee", "rrai": "ai", "rro": "o", "rroo": "oo",
    "rrow": "au", "rrm": "am", "rrah": "ah", "rru": "u", "rruu": "uu",
    # Specific folder-alias combinations
    "an": "none", "ana": "aa", "ani": "i", "anii": "ii", "anu": "u", "anuu": "uu",
    "ane": "e", "anee": "ee", "anai": "ai", "ano": "o", "anoo": "oo", "anou": "au",
    "anm": "am", "anah": "ah", "anr": "ru", "anru": "ru",
    "bm": "am", "bah": "ah", "bhm": "am", "bhah": "ah",
    "chm": "am", "chah": "ah", "chr": "ru", "chru": "ru", "chuu": "uu", "chow": "au",
    "dah": "ah", "dm": "am", "dru": "ru", "druu": "ruu",
    "dhah": "ah", "dhm": "am", "dhru": "ru", "dhruu": "ruu",
    "gah": "ah", "gm": "am", "gru": "ru", "gruu": "ruu",
    "ghah": "ah", "ghm": "am", "ghru": "ru", "ghruu": "ruu",
    "hah": "ah", "hm": "am", "hru": "ru", "hruu": "ruu",
    "jah": "ah", "jm": "am", "jru": "ru", "jruu": "ruu",
    "jhah": "ah", "jhm": "am", "jhru": "ru", "jhruu": "ruu",
    "kah": "ah", "km": "am", "kru": "ru", "kruu": "ruu",
    "khah": "ah", "khm": "am", "khru": "ru", "khruu": "ruu",
    "lah": "ah", "lm": "am", "lru": "ru", "lruu": "ruu",
    "llah": "ah", "llm": "am", "llru": "ru", "llruu": "ruu",
    "mah": "ah", "mm": "am", "mru": "ru", "mruu": "ruu",
    "nah": "ah", "nm": "am", "nru": "ru", "nruu": "ruu",
    "pah": "ah", "pm": "am", "pru": "ru", "pruu": "ruu",
    "phah": "ah", "phm": "am", "phru": "ru", "phruu": "ruu",
    "rah": "ah", "rm": "am",
    "rrah": "ah",
    "sah": "ah", "sm": "am", "sru": "ru", "sruu": "ruu",
    "shah": "ah", "shm": "am", "shru": "ru", "shruu": "ruu",
    "tah": "ah", "tm": "am", "tru": "ru", "truu": "ruu",
    "thah": "ah", "thm": "am", "thru": "ru", "thruu": "ruu",
    "vah": "ah", "vm": "am", "vru": "ru", "vruu": "ruu",
    "yah": "ah", "ym": "am", "yru": "ru", "yruu": "ruu",
    "zh": "ah", "zm": "am", "zru": "ru", "zruu": "ruu", "z": "none"
}

VATTU_ALIASES = {
    "an": "ana", "ana": "ana", "nn": "ana",
    "b": "b", "ba": "b", "bh": "bh", "bha": "bh",
    "c": "c", "ca": "c", "ch": "ch", "cha": "ch", "chh": "ch",
    "d": "d", "da": "d_dental", "dh": "dh", "dha": "dh_dental",
    "g": "g", "ga": "g", "gh": "gh", "gha": "gh",
    "h": "h", "ha": "h",
    "j": "j", "ja": "j", "jh": "jh", "jha": "jh",
    "k": "k", "ka": "k", "kh": "kh", "kha": "kh", "ksh": "ks", "ks": "ks",
    "l": "l", "la": "l", "ll": "ll", "lla": "ll",
    "m": "m", "ma": "m",
    "n": "n", "na": "n",
    "p": "p", "pa": "p", "P": "p", "ph": "ph", "pha": "ph", "Ph": "ph",
    "r": "r", "ra": "r", "rr": "rr", "rra": "rr", "RR": "rr",
    "s": "s", "sa": "s", "sh": "sh", "sha": "sha",
    "t": "t", "ta": "t", "tt": "t", "th": "th", "tha": "th_dental", "th_dental": "th_dental",
    "v": "v", "va": "v",
    "y": "y", "ya": "y",
    "z": "z"
}


def _lookup_alias(table: Dict[str, str], key: str, default: Optional[str] = None) -> str:
    """Performs alias lookup with case-insensitive fallback."""
    if key in table:
        return table[key]
    key_lower = key.lower()
    if key_lower in table:
        return table[key_lower]
    if default is not None:
        return default
    raise KeyError(f"Key '{key}' (or lowercase '{key_lower}') not found in alias table")


def parse_gunintham_modifier(c_key: str, v_key: str) -> str:
    """Parses the vowel modifier from Guninthamulu folder names with exact suffix resolution."""
    c = c_key.lower()
    v = v_key.lower()
    
    # 1. Visarga (ah)
    if v.endswith('aha') or v.endswith('ah') or v in ('gaha', 'ghaha', 'kaha', 'khaha', 'taha', 'rrah', 'anah', 'bah', 'bhah', 'chah', 'dah', 'dhah', 'hah', 'jah', 'jhah', 'kshah', 'lah', 'llah', 'mah', 'nah', 'pah', 'phah', 'rah', 'sah', 'shah', 'thah', 'vah', 'yah', 'zh'):
        return 'ah'
        
    # 2. Sunna (am)
    if v.endswith('am') or (v.endswith('m') and v not in ('m', 'rm', 'rrm')) or v in (
        'anm', 'bm', 'bhm', 'chm', 'dm', 'dhm', 'gm', 'ghm', 'hm', 'jm', 'jhm', 
        'km', 'khm', 'ksham', 'lm', 'llm', 'mm', 'nm', 'pm', 'phm', 'rm', 'rrm', 
        'sm', 'shm', 'tm', 'thm', 'vm', 'ym', 'zm'
    ):
        return 'am'
        
    # 3. Autwam (au)
    if v.endswith('au') or v.endswith('ou') or v.endswith('ow') or v in (
        'anou', 'rrow', 'chow', 'how', 'thow', 'vow', 'you', 'tou', 'kou', 'khou', 
        'gou', 'ghou', 'jou', 'jhou', 'dou', 'dhou', 'pou', 'mou', 'nou', 'lou', 
        'rou', 'sou', 'shou'
    ):
        return 'au'
        
    # 4. Aitwam (ai)
    if v.endswith('ai') or v in ('anai', 'rrai'):
        return 'ai'
        
    # Explicit folder maps for consonants with irregular naming patterns
    if c == 'rr':
        rr_map = {'rr': 'none', 'rra': 'aa', 'rri': 'i', 'rrii': 'ii', 'rru': 'u', 'rruu': 'uu', 'r': 'ru', 'rrr': 'ruu', 'rre': 'e', 'rree': 'ee', 'rrai': 'ai', 'rro': 'o', 'rroo': 'oo', 'rrow': 'au', 'rrm': 'am', 'rrah': 'ah'}
        if v in rr_map:
            return rr_map[v]
            
    if c == 'ra':
        ra_map = {'r': 'none', 'ra': 'aa', 'ri': 'i', 'rii': 'ii', 'ru': 'u', 'ruu': 'uu', 'rr': 'ru', 'rru': 'ruu', 're': 'e', 'ree': 'ee', 'rai': 'ai', 'ro': 'o', 'roo': 'oo', 'rou': 'au', 'rm': 'am', 'rah': 'ah'}
        if v in ra_map:
            return ra_map[v]
            
    if c == 'cha':
        cha_map = {'ch': 'none', 'cha': 'aa', 'chi': 'i', 'chii': 'ii', 'chu': 'u', 'chuu': 'uu', 'chru': 'ru', 'chruu': 'ruu', 'che': 'e', 'chee': 'ee', 'chai': 'ai', 'cho': 'o', 'choo': 'oo', 'chow': 'au', 'chm': 'am', 'chah': 'ah'}
        if v in cha_map:
            return cha_map[v]
        
    # 5. Vatrusudi Dirgham (ruu)
    if v in ('rrr', 'rruu', 'druu', 'dhruu', 'gruu', 'ghruu', 'hruu', 'jruu', 'jhruu', 'kruu', 'khruu', 'kshruu', 'lruu', 'llruu', 'mruu', 'nruu', 'pruu', 'phruu', 'sruu', 'shruu', 'truu', 'thruu', 'vruu', 'yruu', 'bruu', 'bhruu', 'chruu'):
        return 'ruu'
    if c in ('an', 'ana', 'ch', 'dh', 'dha', 'ja', 'jh', 'ksh', 'l', 'll', 'ma', 'na', 'pa', 'pha', 'sa', 'sh', 'sha', 'th', 'tha', 'tt', 'va', 'ya') and v in ('anru', 'chru', 'dhru', 'jru', 'jhru', 'kshru', 'lru', 'llru', 'mru', 'nru', 'pru', 'phru', 'sru', 'shru', 'thru', 'tru', 'vru', 'yru'):
        return 'ruu'
    if c == 'd' and v == 'dru':
        return 'ruu'
    if v.endswith('ruu'):
        return 'ruu'
        
    # 6. Vatrusudi (ru)
    if v in (
        'r', 'anr', 'chr', 'dr', 'dhr', 'jr', 'jhr', 'kshr', 
        'lr', 'mr', 'nr', 'pr', 'sr', 'shr', 'tr', 'thr', 'vr', 'yr', 
        'bru', 'bhru', 'gru', 'ghru', 'hru', 'kru', 'khru', 'dru'
    ):
        return 'ru'
    if c in ('ba', 'bha', 'ga', 'gha', 'ha', 'kha', 'khh', 'ta', 'da') and v in ('bru', 'bhru', 'gru', 'ghru', 'hru', 'kru', 'khru', 'tru', 'dru'):
        return 'ru'
    if c in ('an', 'ana', 'ch', 'dh', 'dha', 'ja', 'jh', 'ksh', 'l', 'll', 'ma', 'na', 'pa', 'pha', 'sa', 'sh', 'sha', 'th', 'tha', 'tt', 'va', 'ya') and v in ('anr', 'chr', 'dhr', 'jr', 'jhr', 'kshr', 'lr', 'mr', 'nr', 'pr', 'sr', 'shr', 'thr', 'tr', 'vr', 'yr'):
        return 'ru'
    if v.endswith('ru') and v not in ('anu', 'chu', 'kshu'):
        return 'ru'
        
    # 7. Gudi Dirgham (ii)
    if v.endswith('ii') or v in ('rrii', 'anii'):
        return 'ii'
        
    # 8. Gudi (i)
    if v.endswith('i') or v in ('rri', 'ani'):
        return 'i'
        
    # 9. Kommu Dirgham (uu)
    if v.endswith('uu') or v in ('rruu', 'anuu', 'chuu', 'kshuu', 'buu', 'bhuu', 'duu', 'guu', 'ghuu', 'huu', 'juu', 'jhuu', 'kuu', 'khuu', 'luu', 'muu', 'nuu', 'puu', 'ruu', 'suu', 'shuu', 'tuu', 'vuu', 'yuu'):
        return 'uu'
        
    # 10. Kommu (u)
    if v.endswith('u') or v in ('rru', 'anu', 'chu', 'kshu', 'bu', 'bhu', 'du', 'gu', 'ghu', 'hu', 'ju', 'jhu', 'ku', 'khu', 'lu', 'mu', 'nu', 'pu', 'ru', 'su', 'shu', 'tu', 'vu', 'yu'):
        return 'u'
        
    # 11. Etwam Dirgham (ee)
    if v.endswith('ee') or v in ('rree', 'anee'):
        return 'ee'
        
    # 12. Etwam (e)
    if v.endswith('e') or v in ('rre', 'ane'):
        return 'e'
        
    # 13. Otwam Dirgham (oo)
    if v.endswith('oo') or v in ('rroo', 'anoo', 'yoo'):
        return 'oo'
        
    # 14. Otwam (o)
    if v.endswith('o') or v in ('rro', 'ano'):
        return 'o'
        
    # 15. Base Talakattu ('none') vs Dirgham ('aa')
    double_a_bases = {'ga': ('ga', 'gaa'), 'gha': ('gha', 'ghaa'), 'kha': ('ka', 'kaa'), 'khh': ('kha', 'khaa'), 'ta': ('ta', 'taa'), 'da': ('da', 'daa')}
    if c in double_a_bases:
        base_v, dirgham_v = double_a_bases[c]
        if v == base_v:
            return 'none'
        if v == dirgham_v:
            return 'aa'
            
    # Folders where shorter name is 'none' and longer name with 'a' is 'aa'
    if v in ('b', 'bh', 'ch', 'd', 'dh', 'h', 'j', 'jh', 'k', 'kh', 'ksh', 'l', 'm', 'n', 'p', 'r', 'rr', 's', 'sh', 't', 'th', 'v', 'y', 'an'):
        return 'none'
        
    if v in ('ba', 'bha', 'cha', 'da', 'dha', 'ha', 'ja', 'jha', 'ka', 'kha', 'ksha', 'la', 'ma', 'na', 'pa', 'ra', 'rra', 'sa', 'sha', 'ta', 'tha', 'va', 'ya', 'ana'):
        return 'aa'
        
    if v_key in ('RRA',):
        return 'aa'
        
    if v.endswith('aa'):
        return 'aa'
        
    return 'none'


def decompose_class_name(class_name: str) -> Tuple[str, str, str]:
    """Decomposes any canonical dataset class name into (base_letter, vowel_modifier, vattu_modifier).
    
    Args:
        class_name: String identifier, e.g. 'Guninthamulu__kha__ki', 'hallulu__ka', 'achulu__a', 'othulu__v'
        
    Returns:
        (base_letter, vowel_modifier, vattu_modifier) as strings.
    """
    parts = class_name.replace("/", "__").replace("\\", "__").split("__")
    cat = parts[0].lower()
    
    if cat == "achulu":
        v_key = parts[1] if len(parts) > 1 else "a"
        base_letter = _lookup_alias(VOWEL_ALIASES, v_key)
        modifier = "none"
        vattu = "none"
        
    elif cat == "hallulu":
        c_key = parts[1] if len(parts) > 1 else "ka"
        base_letter = _lookup_alias(CONSONANT_ALIASES, c_key)
        modifier = "none"
        vattu = "none"
        
    elif cat == "guninthamulu":
        c_key = parts[1] if len(parts) > 1 else "ka"
        v_key = parts[2] if len(parts) > 2 else "a"
        base_letter = _lookup_alias(CONSONANT_ALIASES, c_key)
        modifier = parse_gunintham_modifier(c_key, v_key)
        vattu = "none"
        
    elif cat == "othulu":
        c_key = parts[1] if len(parts) > 1 else "k"
        base_letter = _lookup_alias(CONSONANT_ALIASES, c_key)
        modifier = "none"
        vattu = _lookup_alias(VATTU_ALIASES, c_key)
        
    else:
        raise ValueError(f"Unknown category prefix '{cat}' in class name '{class_name}'")
        
    return base_letter, modifier, vattu


def build_and_validate_label_maps(class_names: List[str], 
                                  output_path: Optional[str] = None) -> Dict[str, Any]:
    """Empirically validates all class names in the dataset and constructs label mappings.
    
    Args:
        class_names: Full list of unique class directory names across the entire dataset.
        output_path: Optional path to save outputs/label_maps.json.
        
    Returns:
        Dictionary containing derived primitive lists, index mappings, valid triples, and reverse lookup index.
    """
    decomposed_records = []
    base_set = set()
    mod_set = set()
    vattu_set = set()
    
    for cname in class_names:
        base, mod, vattu = decompose_class_name(cname)
        decomposed_records.append((cname, base, mod, vattu))
        base_set.add(base)
        mod_set.add(mod)
        vattu_set.add(vattu)
        
    # Derive canonical sorted lists (preserve standard ordering for known primitives)
    base_letters = [b for b in CANONICAL_BASE_LETTERS if b in base_set]
    base_letters += sorted(list(base_set - set(base_letters)))
    
    vowel_modifiers = [m for m in CANONICAL_VOWEL_MODIFIERS if m in mod_set]
    vowel_modifiers += sorted(list(mod_set - set(vowel_modifiers)))
    
    conjunct_modifiers = [v for v in CANONICAL_CONJUNCT_MODIFIERS if v in vattu_set]
    conjunct_modifiers += sorted(list(vattu_set - set(conjunct_modifiers)))
    
    base_map = {b: i for i, b in enumerate(base_letters)}
    mod_map = {m: i for i, m in enumerate(vowel_modifiers)}
    vattu_map = {v: i for i, v in enumerate(conjunct_modifiers)}
    
    combination_to_class: Dict[str, str] = {}
    class_to_combination: Dict[str, str] = {}
    valid_triples_set = set()
    valid_triples: List[List[int]] = []
    
    for cname, base, mod, vattu in decomposed_records:
        b_idx = base_map[base]
        m_idx = mod_map[mod]
        v_idx = vattu_map[vattu]
        key = f"{b_idx}_{m_idx}_{v_idx}"
        class_to_combination[cname] = key
        if key not in combination_to_class:
            combination_to_class[key] = cname
        if (b_idx, m_idx, v_idx) not in valid_triples_set:
            valid_triples_set.add((b_idx, m_idx, v_idx))
            valid_triples.append([b_idx, m_idx, v_idx])
            
    label_maps = {
        "num_base_classes": len(base_letters),
        "num_modifier_classes": len(vowel_modifiers),
        "num_vattu_classes": len(conjunct_modifiers),
        "num_unique_combinations": len(valid_triples),
        "base_letters": base_letters,
        "base_map": base_map,
        "vowel_modifiers": vowel_modifiers,
        "mod_map": mod_map,
        "conjunct_modifiers": conjunct_modifiers,
        "vattu_map": vattu_map,
        "combination_to_class": combination_to_class,
        "class_to_combination": class_to_combination,
        "valid_triples": valid_triples,
        "total_classes": len(class_names)
    }
    
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(label_maps, f, ensure_ascii=False, indent=2)
            
    return label_maps


def recombine_prediction(base_probs: np.ndarray,
                         mod_probs: np.ndarray,
                         vattu_probs: np.ndarray,
                         label_maps: Dict[str, Any]) -> Dict[str, Any]:
    """Recombines multi-head probability distributions into 630-way class predictions.
    
    Uses:
      1. Fast greedy argmax lookup
      2. Constrained Maximum Likelihood Decoding over valid dataset triples if greedy is invalid
      3. Top-5 joint ranking calculation
      
    Args:
        base_probs: 1D probability array of shape (num_base,)
        mod_probs: 1D probability array of shape (num_mod,)
        vattu_probs: 1D probability array of shape (num_vattu,)
        label_maps: Dictionary loaded from label_maps.json
        
    Returns:
        Dictionary with predicted class, components, confidence score, fallback flag, and top-5 rankings.
    """
    base_probs = np.asarray(base_probs, dtype=np.float64)
    mod_probs = np.asarray(mod_probs, dtype=np.float64)
    vattu_probs = np.asarray(vattu_probs, dtype=np.float64)
    
    # Clip probabilities for stable log calculation
    eps = 1e-12
    p_b_safe = np.clip(base_probs, eps, 1.0)
    p_m_safe = np.clip(mod_probs, eps, 1.0)
    p_v_safe = np.clip(vattu_probs, eps, 1.0)
    
    log_p_b = np.log(p_b_safe)
    log_p_m = np.log(p_m_safe)
    log_p_v = np.log(p_v_safe)
    
    comb_map = label_maps["combination_to_class"]
    valid_triples = label_maps["valid_triples"]
    base_letters = label_maps["base_letters"]
    vowel_modifiers = label_maps["vowel_modifiers"]
    conjunct_modifiers = label_maps["conjunct_modifiers"]
    
    # Greedy Check
    b_star = int(np.argmax(base_probs))
    m_star = int(np.argmax(mod_probs))
    v_star = int(np.argmax(vattu_probs))
    greedy_key = f"{b_star}_{m_star}_{v_star}"
    
    if greedy_key in comb_map:
        pred_class = comb_map[greedy_key]
        pred_b, pred_m, pred_v = b_star, m_star, v_star
        confidence = float(base_probs[b_star] * mod_probs[m_star] * vattu_probs[v_star])
        is_fallback = False
    else:
        # Constrained Maximum Likelihood Decoding over all valid triples
        best_score = -np.inf
        best_triple = valid_triples[0]
        for b, m, v in valid_triples:
            score = log_p_b[b] + log_p_m[m] + log_p_v[v]
            if score > best_score:
                best_score = score
                best_triple = [b, m, v]
        pred_b, pred_m, pred_v = best_triple
        pred_class = comb_map[f"{pred_b}_{pred_m}_{pred_v}"]
        confidence = float(np.exp(best_score))
        is_fallback = True
        
    # Top-5 joint scoring across all valid triples
    all_scores = []
    for b, m, v in valid_triples:
        joint_prob = float(base_probs[b] * mod_probs[m] * vattu_probs[v])
        cname = comb_map[f"{b}_{m}_{v}"]
        all_scores.append({
            "class_name": cname,
            "base_letter": base_letters[b],
            "vowel_modifier": vowel_modifiers[m],
            "vattu": conjunct_modifiers[v],
            "probability": joint_prob,
            "indices": (b, m, v)
        })
    all_scores.sort(key=lambda x: x["probability"], reverse=True)
    top_5 = all_scores[:5]
    
    return {
        "predicted_class": pred_class,
        "base_letter": base_letters[pred_b],
        "vowel_modifier": vowel_modifiers[pred_m],
        "vattu": conjunct_modifiers[pred_v],
        "base_idx": pred_b,
        "modifier_idx": pred_m,
        "vattu_idx": pred_v,
        "confidence": confidence,
        "is_fallback": is_fallback,
        "top_5": top_5
    }
