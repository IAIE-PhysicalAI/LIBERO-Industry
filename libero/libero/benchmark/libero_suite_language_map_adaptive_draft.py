"""Draft ambiguity-adaptive language layer for LIBERO-Industry.

This module is intentionally not imported by ``benchmark.__init__``.  It keeps
the production map unchanged while the prompt wording and translations are
audited.  The existing map contains one canonical English prompt per task;
this draft shows how to add controlled variants without changing task meaning.
"""

from .libero_suite_language_map import libero_language_map as _production_language_map


# Copy instead of mutating the production dictionary.
libero_language_map = dict(_production_language_map)


LANGUAGE_POLICY = {
    "geometric": {
        "description": "Target and destination are uniquely determined by visible geometry.",
        "allowed_variation": "verb, syntax, and non-critical synonyms",
        "protected_semantics": ("target_count", "geometric_relation", "destination_type"),
    },
    "attribute_specific": {
        "description": "Distractors require intrinsic object attributes for disambiguation.",
        "allowed_variation": "verb and syntax only",
        "protected_semantics": ("object_class", "color", "shape", "destination_type"),
    },
    "relation_specific": {
        "description": "Similar objects or repeated fixtures require an explicit assignment.",
        "allowed_variation": "syntax only",
        "protected_semantics": (
            "object_identity",
            "target_count",
            "left_right",
            "near_far",
            "destination_type",
        ),
    },
}


# Representative entries only.  Extend this table after semantic and visual
# review; do not generate training labels from an unreviewed translation.
LANGUAGE_VARIANTS = {
    "D2S_LV3_001_SCHOOL_BUS_MINI_ROLLER_MARIO_V3_sort_toys_to_inlays": {
        "ambiguity": "geometric",
        "train": {
            "en": (
                "Place each unmatched toy into its shape-matching inlay.",
                "Fit the remaining toys into their corresponding inlays.",
            ),
            "zh": (
                "将每个尚未归位的玩具放入与其形状匹配的嵌槽中。",
                "将剩余玩具分别嵌入对应的凹槽中。",
            ),
            "de": (
                "Setze jedes noch nicht zugeordnete Spielzeug in die passende Aussparung.",
                "Füge die übrigen Spielzeuge in ihre entsprechenden Aussparungen ein.",
            ),
        },
        "eval_holdout": {
            "en": "Match every remaining toy to the inlay with the same shape.",
            "zh": "把所有剩余玩具放入形状相符的嵌槽。",
            "de": "Ordne jedes verbleibende Spielzeug der formgleichen Aussparung zu.",
        },
    },
    "D2S_LV2_007_THOMAS_PORTER_SCHOOL_BUS_V5_sort_marked_toys_to_grooves": {
        "ambiguity": "relation_specific",
        "train": {
            "en": (
                "Place the blue toy train in the left groove, the gray toy train in the center groove, and the yellow school bus in the right groove.",
            ),
            "zh": (
                "将蓝色玩具火车放入左侧凹槽，将灰色玩具火车放入中间凹槽，并将黄色校车放入右侧凹槽。",
            ),
            "de": (
                "Setze den blauen Spielzeugzug in die linke Aussparung, den grauen Spielzeugzug in die mittlere Aussparung und den gelben Schulbus in die rechte Aussparung.",
            ),
        },
        "eval_holdout": {},
        "review_status": "requires rendered-scene and native-speaker verification",
    },
}


assert set(LANGUAGE_VARIANTS).issubset(libero_language_map)
