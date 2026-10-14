#!/usr/bin/env python3
"""Show every conveyor-belt placement region in MuJoCo's viewer."""

import argparse
from pathlib import Path

import mujoco


REPO_ROOT = Path(__file__).resolve().parents[1]
CONVEYOR_XML = (
    REPO_ROOT
    / "libero"
    / "libero"
    / "assets"
    / "industry_objects"
    / "conveyor_belt"
    / "conveyor_belt.xml"
)
COLORS = (
    (0.15, 0.85, 0.25, 0.40),
    (0.15, 0.45, 1.00, 0.40),
    (1.00, 0.75, 0.10, 0.40),
    (0.95, 0.20, 0.80, 0.40),
    (0.20, 0.95, 0.95, 0.40),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Print regions and validate the XML without opening the viewer.",
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(CONVEYOR_XML))
    data = mujoco.MjData(model)
    regions = []
    for site_id in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
        if name and ("region" in name or "bowl" in name):
            model.site_rgba[site_id] = COLORS[len(regions) % len(COLORS)]
            regions.append(name)
            print(
                "{:<28} pos={} size={} quat={}".format(
                    name,
                    model.site_pos[site_id].round(5).tolist(),
                    model.site_size[site_id].round(5).tolist(),
                    model.site_quat[site_id].round(5).tolist(),
                )
            )

    if not regions:
        raise RuntimeError("No conveyor regions found in {}".format(CONVEYOR_XML))
    print("\n{} visible conveyor regions".format(len(regions)))
    if args.check_only:
        return

    from mujoco import viewer

    viewer.launch(model, data)


if __name__ == "__main__":
    main()
