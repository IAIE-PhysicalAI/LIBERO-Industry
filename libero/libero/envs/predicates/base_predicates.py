from typing import List
import numpy as np          
import robosuite.utils.transform_utils as T
import inspect
from libero.libero.assets.industry_objects.cosmetics_inlay_orientation import (
    cosmetics_inlay_alignment,
)

class Expression:
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class UnaryAtomic(Expression):
    def __init__(self):
        pass

    def __call__(self, arg1):
        raise NotImplementedError


class BinaryAtomic(Expression):
    def __init__(self):
        pass

    def __call__(self, arg1, arg2):
        raise NotImplementedError


class MultiarayAtomic(Expression):
    def __init__(self):
        pass

    def __call__(self, *args):
        raise NotImplementedError


class TruePredicateFn(MultiarayAtomic):
    def __init__(self):
        super().__init__()

    def __call__(self, *args):
        return True


class FalsePredicateFn(MultiarayAtomic):
    def __init__(self):
        super().__init__()

    def __call__(self, *args):
        return False


class InContactPredicateFn(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return arg1.check_contact(arg2)


class In(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return arg2.check_contact(arg1) and arg2.check_contain(arg1)


class On(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return arg2.check_ontop(arg1)

        # if arg2.object_state_type == "site":
        #     return arg2.check_ontop(arg1)
        # else:
        #     obj_1_pos = arg1.get_geom_state()["pos"]
        #     obj_2_pos = arg2.get_geom_state()["pos"]
        #     # arg1.on_top_of(arg2) ?
        #     # TODO (Yfeng): Add checking of center of mass are in the same regions
        #     if obj_1_pos[2] >= obj_2_pos[2] and arg2.check_contact(arg1):
        #         return True
        #     else:
        #         return False


class Up(BinaryAtomic):
    def __call__(self, arg1):
        return arg1.get_geom_state()["pos"][2] >= 1.0


class Stack(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return (
            arg1.check_contact(arg2)
            and arg2.check_contain(arg1)
            and arg1.get_geom_state()["pos"][2] > arg2.get_geom_state()["pos"][2]
        )


class PrintJointState(UnaryAtomic):
    """This is a debug predicate to allow you print the joint values of the object you care"""

    def __call__(self, arg):
        print(arg.get_joint_state())
        return True


class Open(UnaryAtomic):
    def __call__(self, arg):
        return arg.is_open()


class Close(UnaryAtomic):
    def __call__(self, arg):
        return arg.is_close()


class TurnOn(UnaryAtomic):
    def __call__(self, arg):
        return arg.turn_on()


class TurnOff(UnaryAtomic):
    def __call__(self, arg):
        return arg.turn_off()

class AtSpeed(MultiarayAtomic):
    """
    Records the operational state of a conveyor.
    Usage in BDDL: (AtSpeed conveyor_belt_1 0.05 steady)
    """
    def __call__(self, *args):
        return True

class LoadSlot(MultiarayAtomic):
    """
    Records the operational state of a conveyor.
    Usage in BDDL: (LoadSlot conveyor_belt_1 0.05 steady)
    """
    def __call__(self, *args):
        return True

def _check_contact_state(env, obj_name, target_names):
    """
    Scans the contact array to determine if the object is touching any of the targets
    and whether it is currently being held by the gripper.
    Normalized to lowercase for robust collision detection in dynamic environments.
    """
    is_touching = False
    is_held = False
    
    obj_lower = obj_name.lower()
    targets_lower = [t.lower() for t in target_names]
    
    for i in range(env.sim.data.ncon):
        c = env.sim.data.contact[i]
        
        # Get IDs and Names, defaulting to empty string if None
        b1_id = env.sim.model.geom_bodyid[c.geom1]
        b2_id = env.sim.model.geom_bodyid[c.geom2]
        
        b1_name = (env.sim.model.body_id2name(b1_id) or "").lower()
        b2_name = (env.sim.model.body_id2name(b2_id) or "").lower()
        g1_name = (env.sim.model.geom_id2name(c.geom1) or "").lower()
        g2_name = (env.sim.model.geom_id2name(c.geom2) or "").lower()

        # If the contact involves our target object
        if obj_lower in b1_name or obj_lower in b2_name:
            # 1. Gripper check
            if any(x in name for name in (b1_name, b2_name, g1_name, g2_name) for x in ("finger", "gripper")):
                is_held = True
                
            # 2. Target Contact check
            if any(t in b1_name or t in b2_name for t in targets_lower):
                is_touching = True
                
    return is_touching, is_held

def _get_relative_pose(env, obj_name, site_name):
    """
    Uses transform_utils to compute the 4x4 homogeneous relative pose of the object
    in the site's local coordinate frame.
    Returns: (rel_pose, site_id)
    """
    obj_body_id = None
    for suffix in ["_main", "_root", ""]:
        try:
            obj_body_id = env.sim.model.body_name2id(f"{obj_name}{suffix}")
            break
        except ValueError:
            pass
            
    if obj_body_id is None:
        raise ValueError(f"Body for {obj_name} not found")
        
    site_id = env.sim.model.site_name2id(site_name)
    
    # Construct homogeneous poses using transform_utils
    obj_pose = T.make_pose(env.sim.data.body_xpos[obj_body_id], env.sim.data.body_xmat[obj_body_id].reshape(3, 3))
    site_pose = T.make_pose(env.sim.data.site_xpos[site_id], env.sim.data.site_xmat[site_id].reshape(3, 3))
    
    # Compute relative pose: T_Site^Obj = (T_World^Site)^-1 * T_World^Obj
    rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, T.pose_inv(site_pose))
    return rel_pose, site_id

def _get_goal_relative_pose(env, obj_name, site_name):
    """Return pose error in the groove's authored semantic frame when present."""

    rel_pose, site_id = _get_relative_pose(env, obj_name, site_name)
    orientation_name = site_name.replace(
        "_goal_region", "_goal_orientation"
    )
    try:
        orientation_id = env.sim.model.site_name2id(orientation_name)
    except ValueError:
        return rel_pose, site_id, False

    # Flat object assets already contain their canonical flat transform in
    # their collision geoms. Their body therefore remains aligned with the
    # horizontal goal frame instead of applying the transform a second time.
    if "flat" in obj_name.lower() and "flat_groove" in site_name.lower():
        return rel_pose, site_id, True

    obj_body_id = None
    for suffix in ["_main", "_root", ""]:
        try:
            obj_body_id = env.sim.model.body_name2id(f"{obj_name}{suffix}")
            break
        except ValueError:
            pass
    if obj_body_id is None:
        raise ValueError(f"Body for {obj_name} not found")

    obj_pose = T.make_pose(
        env.sim.data.body_xpos[obj_body_id],
        env.sim.data.body_xmat[obj_body_id].reshape(3, 3),
    )
    orientation_pose = T.make_pose(
        env.sim.data.site_xpos[orientation_id],
        env.sim.data.site_xmat[orientation_id].reshape(3, 3),
    )
    semantic_pose = T.pose_in_A_to_pose_in_B(
        obj_pose, T.pose_inv(orientation_pose)
    )
    # Containment remains defined in the horizontal goal-region frame. Only
    # orientation is evaluated in the separate semantic frame.
    semantic_pose[:3, 3] = rel_pose[:3, 3]
    return semantic_pose, site_id, True


def _check_alignment(
    obj_name, target_name, rel_pose, semantic_orientation=False
):
    """Pure geometric logic waterfall for alignment evaluation."""
    rel_rot = rel_pose[:3, :3]
    xx, zz = rel_rot[0, 0], rel_rot[2, 2]
    yx, yz, zx, zy = (
        rel_rot[1, 0],
        rel_rot[1, 2],
        rel_rot[2, 0],
        rel_rot[2, 1],
    )
    
    n1, n2 = obj_name.lower(), target_name.lower()

    cosmetics_alignment = cosmetics_inlay_alignment(obj_name, target_name, rel_rot)
    if cosmetics_alignment is not None:
        return cosmetics_alignment

    if semantic_orientation:
        return zz > 0.9 and abs(xx) > 0.9

    if "inverted" in n1 or "inverted" in n2:
        return zz < -0.9 and abs(xx) > 0.9
    if "flat" in n1 and "upright" in n2 and "inlay" in n2:
        # The flat asset's signed contact frame is +90 degrees about body X.
        # Therefore contact-frame == inlay-frame requires this one body frame:
        # X -> +X, Y -> -Z, Z -> +Y. Do not accept its pi-flipped twin.
        return xx > 0.9 and zy < -0.9 and yz > 0.9
    if ("upright" in n1 and "flat" in n2) or ("flat" in n1 and "upright" in n2):
        aligned = zz > 0.9 and abs(xx) > 0.9
        converted = abs(zz) < 0.1 and (abs(zx) > 0.9 or abs(zy) > 0.9)
        return aligned or converted
    if "v_groove" in n2:
        if "cream_jar" in n1:
            # In the flat jar asset the visual lid axis is body -Y. In this
            # site-relative matrix that axis is therefore -column Y.
            semantic_top = -rel_rot[:, 1]
            expected = np.sqrt(0.5)
            return (
                abs(abs(semantic_top[0]) - expected) < 0.15
                and abs(semantic_top[1]) < 0.20
                and abs(semantic_top[2] - expected) < 0.15
            )
        # Accept either V face after physical settling, but never a visibly
        # flat, perpendicular, or vertical pose.
        seat_tilt = abs(float(np.arctan2(zx, zz)))
        min_seat_tilt = 3.0 if "school_bus_flat" in n1 else 12.0
        max_seat_tilt = 75.0 if "school_bus_flat" in n1 else 60.0
        return (
            np.deg2rad(min_seat_tilt) <= seat_tilt <= np.deg2rad(max_seat_tilt)
            and abs(yx) < 0.45
        )
    if "groove" in n2:
        return zz > 0.9 and abs(xx) > 0.75
        
    # Default (Same-pose Inlay)
    return zz > 0.9 and xx > 0.9

class ReleasedOn(BinaryAtomic):
    """
    Evaluates placement precision for generalized tray packing/overlapping.
    Success = Gripper is released AND Object's Center of Mass is within the XY bounds of the tray.
    Direct mesh contact with the fixture is NOT strictly required, allowing objects to rest on each other.
    """
    def __call__(self, arg1, arg2):
        try:
            env = arg1.env
            obj_name = arg1.object_name
            target_name = arg2.object_name
            fixture_name = target_name.replace("_goal_region", "")

            # 1. Gripper Check (Must be fully released)
            _, is_held = _check_contact_state(env, obj_name, [fixture_name, target_name])
            if is_held:
                return False

            # 2. XY Bounding Box Containment Check
            try:
                site_id = env.sim.model.site_name2id(target_name)
            except ValueError:
                # Fallback for classic LIBERO without defined goal region sites.
                # If no site exists, we must fallback to checking physical contact 
                # with either the tray or ANY other object to ensure it didn't fall off the table.
                is_touching, _ = _check_contact_state(env, obj_name, [fixture_name])
                if is_touching:
                    return True
                # Check if touching any other movable object (overlap support)
                for other_obj in env.objects_dict.keys():
                    if other_obj != obj_name:
                        touching_other, _ = _check_contact_state(env, obj_name, [other_obj])
                        if touching_other: return True
                return False

            # 3. Using robosuite transform_utils for exact geometric bounds
            rel_pose, _ = _get_relative_pose(env, obj_name, target_name)
            local_rel_pos = rel_pose[:3, 3]
            site_size = env.sim.model.site_size[site_id]

            # Evaluate only X and Y coordinates; deliberately ignore Z depth
            # This permits the object to be elevated (resting on another toy)
            return (abs(local_rel_pos[0]) <= site_size[0] and 
                    abs(local_rel_pos[1]) <= site_size[1])
            
        except Exception as e:
            print(f"[Predicate Exception Alert] ReleasedOn: {e}") 
            return False

class AllowHoldOn(BinaryAtomic):
    """
    Evaluates placement precision while allowing the gripper to remain grasping.
    Success = Object is physically touching the fixture.
    If a goal region site exists: The object's COM must fall within the X/Y bounds of the site (Z is ignored).
    If no site exists (Classic LIBERO): Fallbacks to pure mesh-to-mesh contact.
    """
    def __call__(self, arg1, arg2):
        try:
            env = arg1.env
            obj_name = arg1.object_name
            target_name = arg2.object_name
            
            fixture_name = target_name.replace("_goal_region", "")

            # 1. Physical Mesh Contact Check (Deliberately ignore gripper state)
            is_touching, _ = _check_contact_state(env, obj_name, [fixture_name, target_name])
            if not is_touching:
                return False

            # 2. Check if a specific goal region site exists in the Mujoco model
            try:
                site_id = env.sim.model.site_name2id(target_name)
            except ValueError:
                # No goal region site exists (e.g., classic LIBERO "plate_1").
                # Mesh contact was confirmed in Step 1, so this is an automatic success.
                return True

            # 3. XY Bounding Box Check (Using robosuite transform_utils)
            rel_pose, _ = _get_relative_pose(env, obj_name, target_name)
            local_rel_pos = rel_pose[:3, 3]
            site_size = env.sim.model.site_size[site_id]

            # Evaluate only X and Y coordinates; deliberately ignore Z depth
            return (abs(local_rel_pos[0]) <= site_size[0] and 
                    abs(local_rel_pos[1]) <= site_size[1])
            
        except Exception as e:
            print(f"[Predicate Exception Alert] AllowHoldOn: {e}") 
            return False

class AlignedOn(BinaryAtomic):
    def __call__(self, arg1, arg2):
        try:
            env = arg1.env
            obj_name = arg1.object_name
            site_name = arg2.object_name
            fixture_name = site_name.replace("_goal_region", "")

            is_touching, is_held = _check_contact_state(env, obj_name, [fixture_name])
            if is_held or not is_touching:
                #if is_held: print(f"[DEBUG] {obj_name} is still being held by gripper.")
                #if not is_touching: print(f"[DEBUG] {obj_name} is NOT touching fixture '{fixture_name}'.")
                return False

            # Extract relative pose using transform_utils
            rel_pose, _, semantic_orientation = _get_goal_relative_pose(
                env, obj_name, site_name
            )

            #print(f"\n[AlignedOn Math Debug] {obj_name.lower()} -> {site_name.lower()}")
            #print(f"   Physical Contact : Confirmed")
            #print(f"   Z-Dot            : {rel_pose[2, 2]:.4f}")
            #print(f"   X-Dot            : {rel_pose[0, 0]:.4f}")

            return _check_alignment(
                obj_name,
                site_name,
                rel_pose,
                semantic_orientation=semantic_orientation,
            )
            
        except Exception as e:
            print(f"\n[CRITICAL ERROR in AlignedOn] {e}")
            return False

class DynamicOn(BinaryAtomic):
    """
    Settled placement predicate for Level 1 trays on a dynamic conveyor.
    Success = Object is flat and supported by the tray + gripper released.
    """
    def __call__(self, arg1, arg2):
        try:
            env = arg1.env
            obj_name = arg1.object_name
            
            if not hasattr(env, 'dynamic_groove_names'):
                return False
                
            # Vectorize the target list to avoid redundant nested loops over the contact array
            tray_names = [g for g in env.dynamic_groove_names if "tray" in g.lower()]
            if not tray_names:
                return False

            tray_locks = getattr(env, '_tray_planar_locks', None)
            if tray_locks is not None:
                lock = tray_locks.get(obj_name)
                if not lock or lock.get('tray') not in tray_names:
                    return False
                _, is_held = _check_contact_state(env, obj_name, [lock['tray']])
                if is_held:
                    try:
                        is_held = env._check_grasp(
                            env.robots[0].gripper,
                            env.objects_dict[obj_name].contact_geoms,
                        )
                    except (AttributeError, KeyError, TypeError):
                        pass
                return not is_held

            # Compatibility fallback for environments without conveyor-slot physics.
            is_touching, is_held = _check_contact_state(env, obj_name, tray_names)
            return is_touching and not is_held
            
        except Exception as e:
            print(f"[Predicate Exception Alert] DynamicOn: {e}")
            return False

class Routed(BinaryAtomic):
    """
    Evaluates if an object is correctly placed in a dynamic Level 2/3 groove.
    Success = contact + CoM containment + alignment + gripper released.
    """
    def __call__(self, arg1, arg2):
        try:
            env = arg1.env
            obj_name = arg1.object_name
            
            if not hasattr(env, 'dynamic_groove_names'):
                return False
                
            for g_name in env.dynamic_groove_names:
                # 1. Bulletproof Contact & Gripper Check
                is_touching, is_held = _check_contact_state(env, obj_name, [g_name])

                pose_lock = getattr(env, "_v_groove_pose_locks", {}).get(
                    obj_name
                )
                locked_to_groove = bool(
                    pose_lock is not None
                    and pose_lock.get("slot") == g_name
                )
                if is_held and locked_to_groove:
                    try:
                        is_held = env._check_grasp(
                            env.robots[0].gripper,
                            env.objects_dict[obj_name].contact_geoms,
                        )
                    except (AttributeError, KeyError, TypeError):
                        pass
                if is_held or (not is_touching and not locked_to_groove):
                    continue

                is_single_carrier = getattr(
                    env, "_is_single_object_carrier", None
                )
                carrier_occupants = getattr(
                    env, "_single_carrier_occupants", None
                )
                if (
                    callable(is_single_carrier)
                    and callable(carrier_occupants)
                    and is_single_carrier(g_name)
                    and carrier_occupants(g_name) != {obj_name}
                ):
                    continue

                routed_in_slot = getattr(env, "_is_routed_in_slot_region", None)
                if callable(routed_in_slot):
                    if routed_in_slot(obj_name, g_name):
                        return True
                    continue

                # Compatibility fallback for environments without slot physics.
                try:
                    site_name = f"{g_name}_goal_region"
                    rel_pose, _ = _get_relative_pose(env, obj_name, site_name)
                    
                    #print(f"\n[Routed Math Debug] {obj_name.lower()} -> {g_name.lower()}")
                    #print(f"   Physical Contact : Confirmed (Gripper Released)")
                    #print(f"   Z-Dot            : {rel_pose[2, 2]:.4f}")
                    #print(f"   X-Dot            : {rel_pose[0, 0]:.4f}")
                    
                    if _check_alignment(obj_name, g_name, rel_pose):
                        #print(f"[DEBUG] Routed: Perfect alignment achieved for {obj_name}.")
                        return True
                        
                except Exception as eval_err:
                    print(f"[DEBUG] Routed Geometric Check Skipped/Failed: {eval_err}")
                    continue
                    
            return False
            
        except Exception as e:
            print(f"[Predicate Exception Alert] Routed: {e}")
            return False
