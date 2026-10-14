"""Pure suite-composition helpers for the modular industry generators.

The legacy generators encode their Level 2 and Level 3 task distributions in
three independent scripts.  This module keeps those distributions explicit
without coupling the sampling code to BDDL or MuJoCo.
"""

from collections import Counter
from itertools import combinations
import random


def asset_counts(assets):
    """Return the asset-count mapping required by ``InitialSceneTemplates``."""
    return dict(Counter(assets))


def named_instances(assets, starting_counts=None):
    """Give each asset occurrence the exact LIBERO ``asset_N`` instance name.

    ``starting_counts`` lets destination objects continue numbering from their
    source-side counterparts, matching the legacy D2D and S2D conventions.
    """
    counts = dict(starting_counts or {})
    instances = []
    for asset in assets:
        counts[asset] = counts.get(asset, 0) + 1
        instances.append(f"{asset}_{counts[asset]}")
    return instances, counts


def instance_asset(instance_name):
    """Remove only the trailing LIBERO instance suffix."""
    asset, separator, suffix = instance_name.rpartition("_")
    if separator and suffix.isdigit():
        return asset
    return instance_name


def strip_pose(asset_name):
    """Remove one supported pose suffix while retaining multiword base names."""
    for suffix in ("_upright", "_flat", "_inverted"):
        if asset_name.endswith(suffix):
            return asset_name[: -len(suffix)]
    return asset_name


def mapped_instances(instances, assets):
    """Select instances matching an asset list without losing duplicates."""
    available = list(instances)
    selected = []
    for asset in assets:
        for index, instance_name in enumerate(available):
            if instance_asset(instance_name) == asset:
                selected.append(instance_name)
                del available[index]
                break
        else:
            raise ValueError("Missing mapped asset instance: {}".format(asset))
    return selected


def remove_assets(assets, to_remove):
    """Remove one occurrence per item, retaining list order and duplicates."""
    remaining = list(assets)
    for asset in to_remove:
        try:
            remaining.remove(asset)
        except ValueError:
            raise ValueError("Mapped asset is not present: {}".format(asset))
    return remaining


def d2s_pose(base_name, pose_cfg, rng=random):
    """Legacy D2S pose rule: Mario is flat; other toys are flat or upright."""
    if base_name in pose_cfg["forced_flat_bases"]:
        return f"{base_name}_flat"
    return f"{base_name}_{rng.choice(pose_cfg['movable_poses'])}"


def d2s_level3_fixture(asset_name):
    """Return the pose-matched static fixture for one Level-3 toy."""
    if asset_name.endswith("_upright"):
        return f"{asset_name}_groove"
    if asset_name.endswith("_flat"):
        return f"{asset_name}_inlay"
    raise ValueError("D2S Level-3 asset has no supported pose: {}".format(asset_name))


def d2d_pose(base_name, pose_cfg, rng=random):
    """Legacy D2D Level 3 rule: every belt object has an inversion roll."""
    pose = "inverted" if rng.random() < pose_cfg["inversion_probability"] else "upright"
    return f"{base_name}_{pose}"


def groove_for_d2s_level2(asset_name, groove_cfg=None, rng=random):
    """Return the track-specific D2S Level-2 groove for an object."""
    if not groove_cfg or groove_cfg.get("match_pose"):
        return f"{asset_name}_groove"
    base_name = strip_pose(asset_name)
    choices = groove_cfg["allowed_by_base"].get(base_name, groove_cfg["default"])
    return f"{base_name}_{rng.choice(choices)}"


def sample_s2d_level2(cfg, target_key, rng=random):
    mapping = cfg["mismatch_map"][target_key]
    composition = cfg["composition"]
    target_assets = [mapping["init"]] * composition["target_count"]
    target_base = strip_pose(target_key)
    candidates = [
        asset for asset in cfg["all_objects"]
        if strip_pose(asset) != target_base
    ]
    source_count = rng.choice(composition["source_confounder_counts"])
    source_confounders = rng.choices(candidates, k=source_count)
    return {
        "target_assets": target_assets,
        "source_confounder_assets": source_confounders,
        "destination_confounder_assets": [],
        "inlay": mapping["inlay"],
    }


def sample_s2d_level3(cfg, target_base, rng=random):
    composition = cfg["composition"]
    target_assets = [
        f"{target_base}_{rng.choice(composition['target_poses'])}"
        for _ in range(composition["target_count"])
    ]
    source_count = rng.choice(composition["source_confounder_counts"])
    source_candidates = [
        asset for asset in cfg["all_objects"]
        if not asset.startswith(f"{target_base}_")
    ]
    source_confounders = rng.choices(source_candidates, k=source_count)

    destination_max = min(
        composition["destination_confounder_max"],
        composition["confounder_total_max"] - source_count,
    )
    destination_count = rng.randint(composition["destination_confounder_min"], destination_max)
    matching_pose = cfg["target_map"][target_base]["pose"]
    destination_confounders = [f"{target_base}_{matching_pose}"] * destination_count
    return {
        "target_assets": target_assets,
        "source_confounder_assets": source_confounders,
        "destination_confounder_assets": destination_confounders,
        "inlay": cfg["target_map"][target_base]["inlay"],
    }


def sample_d2s_level2(cfg, primary_target, rng=random):
    """Sample a D2S Level-2 triple with no more than one huge static groove."""
    composition = cfg["composition"]
    huge_bases = set(composition.get("large_bases", ()))
    others = rng.sample(
        [asset for asset in cfg["all_objects"] if asset != primary_target],
        composition["source_total"] - 1,
    )
    allowed_static = list(others)
    if strip_pose(primary_target) in huge_bases:
        allowed_static = [
            asset for asset in allowed_static
            if strip_pose(asset) not in huge_bases
        ]
    candidate_pairs = [
        pair
        for pair in combinations(allowed_static, composition["goal_groove_count"] - 1)
        if sum(
            strip_pose(asset) in huge_bases
            for asset in (primary_target,) + pair
        ) <= 1
    ]
    if not candidate_pairs:
        raise ValueError("D2S Level-2 has no one-huge static fixture triple")
    groove_targets = [primary_target] + list(rng.choice(candidate_pairs))
    source_assets = [primary_target] + others
    rng.shuffle(source_assets)
    groove_assets = [
        groove_for_d2s_level2(asset, composition["grooves"], rng)
        for asset in groove_targets
    ]
    return {
        "source_assets": source_assets,
        "goal_assets": groove_targets,
        "groove_assets": groove_assets,
        "destination_confounder_assets": [],
    }


def _d2s_candidate_bases(all_bases, target_bases, huge_bases):
    candidates = [base for base in all_bases if base not in target_bases]
    if any(base in huge_bases for base in target_bases):
        candidates = [base for base in candidates if base not in huge_bases]
    return candidates


def sample_d2s_level3(cfg, primary_target, rng=random):
    composition = cfg["composition"]
    huge_bases = set(composition["mutually_exclusive_huge_bases"])
    target_candidates = [base for base in cfg["all_objects"] if base != primary_target]
    if primary_target in huge_bases:
        target_candidates = [base for base in target_candidates if base not in huge_bases]
    target_bases = [primary_target] + rng.sample(
        target_candidates, composition["target_count"] - 1,
    )

    source_count = rng.choice(composition["source_confounder_counts"])
    remaining = _d2s_candidate_bases(cfg["all_objects"], target_bases, huge_bases)
    source_confounder_bases = rng.sample(remaining, source_count)
    destination_candidates = _d2s_candidate_bases(
        cfg["all_objects"], target_bases, huge_bases,
    )
    destination_base = rng.sample(destination_candidates, composition["destination_confounder_count"])[0]
    static_bases = target_bases + [destination_base]
    if sum(base in huge_bases for base in static_bases) > 1:
        raise ValueError("D2S Level-3 sampled more than one huge static fixture")

    source_target_assets = [d2s_pose(base, composition["poses"], rng) for base in target_bases]
    source_confounder_assets = [
        d2s_pose(base, composition["poses"], rng)
        for base in source_confounder_bases
    ]
    source_assets = source_target_assets + source_confounder_assets
    rng.shuffle(source_assets)
    return {
        "target_bases": target_bases,
        "target_assets": source_target_assets,
        "source_confounder_assets": source_confounder_assets,
        "source_assets": source_assets,
        "destination_base": destination_base,
        "destination_confounder_assets": [f"{destination_base}_flat"],
        "target_inlays": [d2s_level3_fixture(asset) for asset in source_target_assets],
        "destination_inlays": [
            d2s_level3_fixture(f"{destination_base}_flat")
        ],
    }


def sample_d2d_level2(cfg, target_asset, rng=random):
    composition = cfg["composition"]
    target_count = rng.choice(composition["target_counts"])
    source_confounder_count = composition["source_total"] - target_count
    confounders = rng.sample(
        [asset for asset in cfg["all_objects"] if asset != target_asset],
        source_confounder_count,
    )
    source_assets = [target_asset] * target_count + confounders
    rng.shuffle(source_assets)
    return {
        "target_assets": [target_asset] * target_count,
        "source_confounder_assets": confounders,
        "source_assets": source_assets,
        "destination_confounder_assets": [],
    }


def sample_d2d_level3(cfg, target_base, rng=random):
    composition = cfg["composition"]
    target_count = rng.choice(composition["target_counts"])
    source_confounder_count = composition["source_total"] - target_count
    source_confounder_bases = rng.choices(
        [base for base in cfg["targets"] if base != target_base],
        k=source_confounder_count,
    )
    target_assets = [d2d_pose(target_base, composition["poses"], rng) for _ in range(target_count)]
    source_confounder_assets = [
        d2d_pose(base, composition["poses"], rng)
        for base in source_confounder_bases
    ]
    source_assets = target_assets + source_confounder_assets
    rng.shuffle(source_assets)
    destination_count = composition["confounder_total"] - source_confounder_count
    destination_assets = [f"{target_base}_upright"] * destination_count
    return {
        "target_assets": target_assets,
        "source_confounder_assets": source_confounder_assets,
        "source_assets": source_assets,
        "destination_confounder_assets": destination_assets,
    }
