from pathlib import Path
from distutils.dir_util import copy_tree

# Copies files from ../../data/HE_defects/*  ->  ../../data/multiclass/
# Grouping classes into superclasses according to a custom-made mapping defined by HE_DEFECTS_2_MULTICLASS_MAIN_MAP

HE_DEFECTS_2_MULTICLASS_MAIN_MAP = {
    "corrosion": [
        "RCo_-_Rusty_Corroded",
        "RSt_-_Rust_stain_streak_spot",
        "RS_-_Corroded_Rusting",
        "RX_-_Corroded_Rusting",
    ],
    "crack": [
        "Cr_-_Crack_of_uncertain_origin_or_a_combination_of_causes",
        "DSCr_-_Drying_shrinkage_crack",
    ],
    "spalling": ["ER_-_Exposed_reinforcement", "Sp_-_Spalled_area"],
}

if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / "data"
    for superclass, subclasses in HE_DEFECTS_2_MULTICLASS_MAIN_MAP.items():
        superclass_dir = data_path / "multiclass_main" / superclass
        superclass_dir.mkdir(parents=True, exist_ok=True)
        for subclass in subclasses:
            copy_tree(
                str(data_path / "HE_defects" / subclass), str(superclass_dir)
            )
