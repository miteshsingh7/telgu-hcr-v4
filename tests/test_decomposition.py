"""Unit tests and full-dataset audit for Telugu decomposition and recombination."""

import json
from pathlib import Path
import numpy as np

from src.data.decomposition import (
    decompose_class_name,
    build_and_validate_label_maps,
    recombine_prediction
)


from src.data.known_duplicates import (
    CONFIRMED_DUPLICATE_CLASSES,
    is_known_duplicate_pair,
    get_equivalent_classes
)


def test_individual_decompositions():
    """Verifies decomposition logic across representative categories and aliases."""
    # Achulu
    assert decompose_class_name("achulu__a") == ("అ", "none", "none")
    assert decompose_class_name("achulu__aa") == ("ఆ", "none", "none")
    assert decompose_class_name("achulu__ah") == ("అః", "none", "none")
    assert decompose_class_name("achulu__ao") == ("ఔ", "none", "none")
    
    # Hallulu
    assert decompose_class_name("hallulu__ka") == ("క", "none", "none")
    assert decompose_class_name("hallulu__P") == ("ప", "none", "none")
    assert decompose_class_name("hallulu__rr") == ("ఱ", "none", "none")
    assert decompose_class_name("hallulu__ta") == ("ట", "none", "none")
    assert decompose_class_name("hallulu__th") == ("ఠ", "none", "none")
    assert decompose_class_name("hallulu__tha") == ("త", "none", "none")
    assert decompose_class_name("hallulu__thah") == ("థ", "none", "none")
    assert decompose_class_name("hallulu__da") == ("ద", "none", "none")
    assert decompose_class_name("hallulu__dha") == ("ధ", "none", "none")
    
    # Guninthamulu
    assert decompose_class_name("Guninthamulu__kha__ki") == ("క", "i", "none")
    assert decompose_class_name("Guninthamulu__khh__khi") == ("ఖ", "i", "none")
    assert decompose_class_name("Guninthamulu__RR__rrah") == ("ఱ", "ah", "none")
    assert decompose_class_name("Guninthamulu__ana__ane") == ("ణ", "e", "none")
    assert decompose_class_name("Guninthamulu__ch__ch") == ("చ", "none", "none")
    assert decompose_class_name("Guninthamulu__cha__ch") == ("ఛ", "none", "none")
    assert decompose_class_name("Guninthamulu__ta__ta") == ("ట", "none", "none")
    assert decompose_class_name("Guninthamulu__tt__t") == ("ఠ", "none", "none")
    assert decompose_class_name("Guninthamulu__th__th") == ("త", "none", "none")
    assert decompose_class_name("Guninthamulu__tha__th") == ("థ", "none", "none")
    assert decompose_class_name("Guninthamulu__d__d") == ("డ", "none", "none")
    assert decompose_class_name("Guninthamulu__da__da") == ("ద", "none", "none")
    assert decompose_class_name("Guninthamulu__dh__dh") == ("ఢ", "none", "none")
    assert decompose_class_name("Guninthamulu__dha__dh") == ("ధ", "none", "none")
    
    # Othulu (isolated subscripts)
    assert decompose_class_name("othulu__v") == ("none", "none", "v")
    assert decompose_class_name("othulu__ks") == ("none", "none", "ks")
    assert decompose_class_name("othulu__an") == ("none", "none", "an")
    assert decompose_class_name("othulu__nn") == ("none", "none", "nn")


def test_full_dataset_decomposition_audit():
    """Scans all class names from the dataset and asserts ZERO unmapped/fallback errors."""
    # Find dataset classes either from raw folders, frozen label_maps.json, or train.csv
    raw_dataset_path = Path("/Users/miteshsingh/Documents/projects/telugu-hcr-v3/data/Final Dataset of Telugu Handwritten Chararcters/Test1")
    lmaps_fallback = Path("outputs/label_maps.json")
    train_csv_fallback = Path("outputs/train.csv")
    
    class_names = []
    if raw_dataset_path.exists():
        for cat_dir in sorted(raw_dataset_path.iterdir()):
            if not cat_dir.is_dir() or cat_dir.name.startswith("."):
                continue
            cat = cat_dir.name
            if cat.lower() == "guninthamulu":
                for c_dir in sorted(cat_dir.iterdir()):
                    if not c_dir.is_dir() or c_dir.name.startswith("."):
                        continue
                    for v_dir in sorted(c_dir.iterdir()):
                        if not v_dir.is_dir() or v_dir.name.startswith("."):
                            continue
                        class_names.append(f"{cat}__{c_dir.name}__{v_dir.name}")
            else:
                for c_dir in sorted(cat_dir.iterdir()):
                    if not c_dir.is_dir() or c_dir.name.startswith("."):
                        continue
                    class_names.append(f"{cat}__{c_dir.name}")
    elif lmaps_fallback.exists():
        with open(lmaps_fallback, "r", encoding="utf-8") as f:
            lm_data = json.load(f)
            class_names = list(lm_data["class_to_combination"].keys())
    elif train_csv_fallback.exists():
        import pandas as pd
        df_train = pd.read_csv(train_csv_fallback)
        class_names = list(df_train["class_name"].unique())
    else:
        raise RuntimeError("Neither raw dataset nor outputs/label_maps.json found for decomposition audit.")
        
    assert len(class_names) >= 500, f"Expected >= 500 classes, found {len(class_names)}"
    
    # Run full decomposition and map generation
    output_map_path = "outputs/label_maps.json"
    label_maps = build_and_validate_label_maps(class_names, output_path=output_map_path)
    
    # Assert empirical counts are populated
    num_base = label_maps["num_base_classes"]
    num_mod = label_maps["num_modifier_classes"]
    num_vattu = label_maps["num_vattu_classes"]
    num_unique = label_maps["num_unique_combinations"]
    
    print(f"Empirically derived vocabulary: num_base={num_base}, num_mod={num_mod}, num_vattu={num_vattu}, unique={num_unique}")
    assert num_base == 52, f"Expected 52 base classes, got {num_base}"
    assert num_mod == 16, f"Expected 16 modifier classes, got {num_mod}"
    assert num_vattu == 37, f"Expected 37 vattu classes, got {num_vattu}"
    assert num_unique == 596, f"Expected 596 unique combinations, got {num_unique}"
    assert len(label_maps["valid_triples"]) == num_unique
    assert len(label_maps["class_to_combination"]) == len(class_names)
    
    # Independent Injectivity Check: Ensure NO unexpected collisions exist
    triple_to_classes = {}
    for cname in class_names:
        triple = decompose_class_name(cname)
        if triple not in triple_to_classes:
            triple_to_classes[triple] = []
        triple_to_classes[triple].append(cname)
        
    # Verify all collisions are in confirmed duplicates list
    for triple, classes in triple_to_classes.items():
        if len(classes) > 1:
            for i in range(len(classes)):
                for j in range(i + 1, len(classes)):
                    c1, c2 = classes[i], classes[j]
                    assert is_known_duplicate_pair(c1, c2), (
                        f"Unexpected collision for triple {triple}: {c1} vs {c2} is not a confirmed duplicate pair!"
                    )
                    
    # Verify all confirmed duplicate pairs map to the EXACT same triple
    for c1, c2 in CONFIRMED_DUPLICATE_CLASSES:
        if c1 in class_names and c2 in class_names:
            t1 = decompose_class_name(c1)
            t2 = decompose_class_name(c2)
            assert t1 == t2, f"Confirmed duplicate pair ({c1}, {c2}) did not map to same triple: {t1} vs {t2}"
            
    # Round-trip check: every class can be decomposed and reconstructed via recombination
    for cname in class_names:
        base, mod, vattu = decompose_class_name(cname)
        b_idx = label_maps["base_map"][base]
        m_idx = label_maps["mod_map"][mod]
        v_idx = label_maps["vattu_map"][vattu]
        
        # Simulate one-hot probability vectors
        b_probs = np.zeros(num_base)
        b_probs[b_idx] = 1.0
        m_probs = np.zeros(num_mod)
        m_probs[m_idx] = 1.0
        v_probs = np.zeros(num_vattu)
        v_probs[v_idx] = 1.0
        
        rec = recombine_prediction(b_probs, m_probs, v_probs, label_maps)
        # Expected canonical reconstructed class
        expected_class = label_maps["combination_to_class"][f"{b_idx}_{m_idx}_{v_idx}"]
        assert rec["predicted_class"] == expected_class
        assert not rec["is_fallback"]
        # Verify the reconstructed class is in the same equivalence group as cname
        assert cname in get_equivalent_classes(rec["predicted_class"])


def test_recombination_constrained_mle_fallback():
    """Verifies that invalid combinations fall back to the most likely valid combination."""
    label_maps = {
        "num_base_classes": 3,
        "num_modifier_classes": 2,
        "num_vattu_classes": 2,
        "base_letters": ["క", "ఖ", "గ"],
        "vowel_modifiers": ["none", "aa"],
        "conjunct_modifiers": ["none", "k"],
        "valid_triples": [
            [0, 0, 0], # క (none, none)
            [0, 1, 0], # కా (aa, none)
            [1, 0, 0]  # ఖ (none, none)
        ],
        "combination_to_class": {
            "0_0_0": "hallulu__ka",
            "0_1_0": "Guninthamulu__ka__kaa",
            "1_0_0": "hallulu__kha"
        }
    }
    
    # Suppose model predicts b=1 (0.6), m=1 (0.8), v=0 (0.9)
    # Triple (1, 1, 0) is INVALID.
    # Scores for valid triples:
    # (0, 0, 0): log(0.3) + log(0.2) + log(0.9) = -1.20 - 1.61 - 0.10 = -2.91
    # (0, 1, 0): log(0.3) + log(0.8) + log(0.9) = -1.20 - 0.22 - 0.10 = -1.52
    # (1, 0, 0): log(0.6) + log(0.2) + log(0.9) = -0.51 - 1.61 - 0.10 = -2.22
    # Winner must be (0, 1, 0)
    b_probs = np.array([0.3, 0.6, 0.1])
    m_probs = np.array([0.2, 0.8])
    v_probs = np.array([0.9, 0.1])
    
    rec = recombine_prediction(b_probs, m_probs, v_probs, label_maps)
    assert rec["predicted_class"] == "Guninthamulu__ka__kaa"
    assert rec["is_fallback"] is True


if __name__ == "__main__":
    test_individual_decompositions()
    test_recombination_constrained_mle_fallback()
    test_full_dataset_decomposition_audit()
    print("All decomposition tests passed successfully!")
