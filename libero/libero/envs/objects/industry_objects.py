import os
import re
import numpy as np

from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import string_to_array

import pathlib

absolute_path = pathlib.Path(__file__).parent.parent.parent.absolute()

from libero.libero.envs.base_object import (
    register_visual_change_object,
    register_object,
)

class IndustryObject(MujocoXMLObject):
    def __init__(self, name, obj_name, joints=[dict(type="free", damping="0.0005")], xml_name=None):
        target_xml = xml_name if xml_name else obj_name
        super().__init__(
            os.path.join(
                str(absolute_path), f"assets/industry_objects/{obj_name}/{target_xml}.xml"
            ),
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.rotation = (0, 0)
        self.rotation_axis = None
        self.object_properties = {
            "vis_site_names": {}
        }
        
        # --- NEW: Automatically extract geom size from XML ---
        geom = self.worldbody.find(".//geom")
        if geom is not None and geom.get("size") is not None:
            # size is usually half-extents: [x, y, z]
            self.parsed_size = string_to_array(geom.get("size"))
        else:
            self.parsed_size = np.array([0.05, 0.05, 0.01]) # Fallback

@register_object
class ConveyorBelt(IndustryObject):
    def __init__(
        self,
        name="conveyor_belt",
        obj_name="conveyor_belt",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {
            "x": (0, 0),
            "y": (0, 0),
            "z": (0, 0),
        }
        self.rotation_axis = None
        self.z_offset = 0.00

# @register_object
# class ConveyorCurved(IndustryObject):
#     def __init__(
#         self,
#         name="conveyor_curved",
#         obj_name="conveyor_curved",
#         joints=[dict(type="free", damping="0.0005")],
#     ):
#         super().__init__(name, obj_name, joints)
#         self.rotation = {
#             "x": (0, 0),
#             "y": (0, 0),
#             "z": (0, 0),
#         }
#         self.rotation_axis = None

@register_object
class ConveyorGhost(IndustryObject):
    def __init__(
        self,
        name="conveyor_ghost",
        obj_name="conveyor_ghost",
        joints=[], 
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {
            "x": (0, 0),
            "y": (0, 0),
            "z": (0, 0),
        }
        self.rotation_axis = None
        self.z_offset = 0.005
        
@register_object
class Box(IndustryObject):
    def __init__(
        self,
        name="box",
        obj_name="box",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {
            "x": (0, 0),
            "y": (0, 0),
            "z": (0, 0),
        }
        self.rotation_axis = None
        self.z_offset = 0.01


@register_object
class ToyTray(IndustryObject):
    def __init__(self, name="toy_tray", obj_name="toy_tray", joints=[dict(type="free", damping="0.0005")]):
        # Pass the specific xml_name to the parent class
        super().__init__(name, obj_name, xml_name="toy_tray", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.005

# cosmetics 
@register_object
class PumpBottleUpright(IndustryObject):
    def __init__(
        self, 
        name="pump_bottle_upright", 
        obj_name="pump_bottle", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="pump_bottle_upright", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = 'z'
        self.z_offset = 0.0609
@register_object
class PumpBottleFlat(IndustryObject):
    def __init__(
        self, 
        name="pump_bottle_flat", 
        obj_name="pump_bottle", 
        joints=[dict(type="free", damping="0.01", frictionloss="0.001")]
    ):
        super().__init__(name, obj_name, xml_name="pump_bottle_flat", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.005
@register_object
class CreamBottleUpright(IndustryObject):
    def __init__(
        self, 
        name="cream_bottle_upright", 
        obj_name="cream_bottle", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="cream_bottle_upright", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = 'z'
        self.z_offset = 0.0459
@register_object
class CreamBottleFlat(IndustryObject):
    def __init__(
        self, 
        name="cream_bottle_flat", 
        obj_name="cream_bottle", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="cream_bottle_flat", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.005
@register_object
class CreamJarUpright(IndustryObject):
    def __init__(
        self, 
        name="cream_jar_upright", 
        obj_name="cream_jar", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="cream_jar_upright", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = 'z'
        self.z_offset = 0.0257
@register_object
class CreamJarFlat(IndustryObject):
    def __init__(
        self, 
        name="cream_jar_flat", 
        obj_name="cream_jar", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="cream_jar_flat", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.005
@register_object
class SerumBottleUpright(IndustryObject):
    def __init__(
        self, 
        name="serum_bottle_upright", 
        obj_name="serum_bottle", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="serum_bottle_upright", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = 'z'
        self.z_offset = 0.0423
@register_object
class SerumBottleFlat(IndustryObject):
    def __init__(
        self, 
        name="serum_bottle_flat", 
        obj_name="serum_bottle", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="serum_bottle_flat", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.005
@register_object
class CosmeticsInlayA(IndustryObject):
    def __init__(
        self,
        name="cosmetics_inlay_a", 
        obj_name="cosmetics_inlay_a",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0  # Half of 0.049

@register_object
class CosmeticsInlayB(IndustryObject):
    def __init__(
        self,
        name="cosmetics_inlay_b", 
        obj_name="cosmetics_inlay_b",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.025

@register_object
class CosmeticsInlayC(IndustryObject):
    def __init__(
        self,
        name="cosmetics_inlay_c", 
        obj_name="cosmetics_inlay_c",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.025

@register_object
class CosmeticsInlayD(IndustryObject):
    def __init__(
        self,
        name="cosmetics_inlay_d", 
        obj_name="cosmetics_inlay_d",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.025

@register_object
class GreyPlate(IndustryObject):
    def __init__(self, name="grey_plate", obj_name="plates", joints=[dict(type="free", damping="0.0005")]):
        # Pass the specific xml_name to the parent class
        super().__init__(name, obj_name, xml_name="grey_plate", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.005

@register_object
class WhitePlate(IndustryObject):
    def __init__(self, name="white_plate", obj_name="plates", joints=[dict(type="free", damping="0.0005")]):
        super().__init__(name, obj_name, xml_name="white_plate", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.005

@register_object
class BlackPlate(IndustryObject):
    def __init__(self, name="black_plate", obj_name="plates", joints=[dict(type="free", damping="0.0005")]):
        super().__init__(name, obj_name, xml_name="black_plate", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.005

# # pump
# @register_object
# class PumpBottleInlayA(IndustryObject):
#     def __init__(
#         self,
#         name="pump_bottle_inlay_a", 
#         obj_name="pump_bottle_inlay_a",
#         joints=[dict(type="free", damping="0.0005")],
#     ):
#         # Override the folder_name to group them together in the assets directory!
#         super().__init__(name, obj_name, joints)
        
#         self.rotation = {
#             "x": (0, 0),
#             "y": (0, 0),
#             "z": (0, 0),
#         }
#         self.rotation_axis = None
#         self.z_offset = 0.00

# @register_object
# class PumpBottleInlayADyn(IndustryObject):
#     def __init__(
#         self,
#         name="pump_bottle_inlay_a_dyn", 
#         obj_name="pump_bottle_inlay_a_dyn",
#         # CHANGE THIS TO SLIDE JOINT ALONG Y-AXIS
#         joints=[dict(type="slide", axis="0 1 0", damping="5.0")],
#     ):
#         super().__init__(name, obj_name, joints)
#         self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
#         self.rotation_axis = None
#         self.z_offset = 0.01 

# @register_object
# class PumpBottleInlayB(IndustryObject):
#     def __init__(
#         self,
#         name="pump_bottle_inlay_b",
#         obj_name="pump_bottle_inlay_b",
#         joints=[dict(type="free", damping="0.0005")],
#     ):
#         super().__init__(name, obj_name, joints)
#         self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
#         self.rotation_axis = None
#         self.z_offset = 0.06


# @register_object
# class PumpBottleInlayC(IndustryObject):
#     def __init__(
#         self,
#         name="pump_bottle_inlay_c",
#         obj_name="pump_bottle_inlay_c",
#         joints=[dict(type="free", damping="0.0005")],
#     ):
#         super().__init__(name, obj_name, joints)
#         self.rotation = {
#             "x": (0, 0),
#             "y": (0, 0),
#             "z": (0, 0)}
#         self.rotation_axis = None
#         self.z_offset = 0.06


# @register_object
# class PumpBottleInlayD(IndustryObject):
#     def __init__(
#         self,
#         name="pump_bottle_inlay_d",
#         obj_name="pump_bottle_inlay_d",
#         joints=[dict(type="free", damping="0.0005")],
#     ):
#         super().__init__(name, obj_name, joints)
#         self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
#         self.rotation_axis = None
#         self.z_offset = 0.08


@register_object
class PumpBottleGroove(IndustryObject):
    def __init__(
        self,
        name="pump_bottle_groove",
        obj_name="pump_bottle_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.05


@register_object
class PumpBottleFlatGroove(IndustryObject):
    def __init__(
        self,
        name="pump_bottle_flat_groove",
        obj_name="pump_bottle_flat_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class PumpBottleFlatVGroove(IndustryObject):
    def __init__(
        self,
        name="pump_bottle_flat_v_groove",
        obj_name="pump_bottle_flat_v_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class CreamBottleFlatGroove(IndustryObject):
    def __init__(
        self,
        name="cream_bottle_flat_groove",
        obj_name="cream_bottle_flat_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class SerumBottleFlatGroove(IndustryObject):
    def __init__(
        self,
        name="serum_bottle_flat_groove",
        obj_name="serum_bottle_flat_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class CreamJarFlatGroove(IndustryObject):
    def __init__(
        self,
        name="cream_jar_flat_groove",
        obj_name="cream_jar_flat_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class CreamJarUprightInlay(IndustryObject):
    def __init__(
        self, 
        name="cream_jar_upright_inlay", 
        obj_name="cream_jar_upright_inlay", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="cream_jar_upright_inlay", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = 'z'
        self.z_offset = 0.0296
@register_object
class CreamBottleFlatInlay(IndustryObject):
    def __init__(
        self, 
        name="cream_bottle_flat_inlay", 
        obj_name="cream_bottle_flat_inlay", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="cream_bottle_flat_inlay", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = 'z'
        self.z_offset = 0.0171
@register_object
class SerumBottleFlatInlay(IndustryObject):
    def __init__(
        self, 
        name="serum_bottle_flat_inlay", 
        obj_name="serum_bottle_flat_inlay", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="serum_bottle_flat_inlay", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = 'z'
        self.z_offset = 0.0186
@register_object
class PumpBottleUprightInlay(IndustryObject):
    def __init__(
        self, 
        name="pump_bottle_upright_inlay", 
        obj_name="pump_bottle_upright_inlay", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="pump_bottle_upright_inlay", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = 'z'
        self.z_offset = 0.0561

@register_object
class CosmeticsTray(IndustryObject):
    def __init__(
        self, 
        name="cosmetics_tray", 
        obj_name="cosmetics_tray", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="cosmetics_tray", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = 'z'
        self.z_offset = 0.0100


@register_object
class CreamJarVGroove(IndustryObject):
    def __init__(
        self,
        name="cream_jar_v_groove",
        obj_name="cream_jar_v_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100




@register_object
class HeadsetUpright(IndustryObject):
    def __init__(
        self, 
        name="headset_upright", 
        obj_name="headset", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="headset_upright", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = 'z'
        self.z_offset = 0.0322

@register_object
class InkCartridgeRedUpright(IndustryObject):
    def __init__(
        self, 
        name="ink_cartridge_red_upright", 
        obj_name="ink_cartridge_red", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="ink_cartridge_red_upright", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = 'z'
        self.z_offset = 0.1146
@register_object
class InkCartridgeRedFlat(IndustryObject):
    def __init__(
        self, 
        name="ink_cartridge_red_flat", 
        obj_name="ink_cartridge_red", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="ink_cartridge_red_flat", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.005
@register_object
class InkCartridgeWhiteUpright(IndustryObject):
    def __init__(
        self, 
        name="ink_cartridge_white_upright", 
        obj_name="ink_cartridge_white", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="ink_cartridge_white_upright", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = 'z'
        self.z_offset = 0.1146
@register_object
class InkCartridgeWhiteFlat(IndustryObject):
    def __init__(
        self, 
        name="ink_cartridge_white_flat", 
        obj_name="ink_cartridge_white", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="ink_cartridge_white_flat", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.005
@register_object
class InkCartridgeGreenUpright(IndustryObject):
    def __init__(
        self, 
        name="ink_cartridge_green_upright", 
        obj_name="ink_cartridge_green", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="ink_cartridge_green_upright", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = 'z'
        self.z_offset = 0.1149
@register_object
class InkCartridgeGreenFlat(IndustryObject):
    def __init__(
        self, 
        name="ink_cartridge_green_flat", 
        obj_name="ink_cartridge_green", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="ink_cartridge_green_flat", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.005


@register_object
class InkCartridgeMagentaUpright(IndustryObject):
    def __init__(
        self, 
        name="ink_cartridge_magenta_upright", 
        obj_name="ink_cartridge_magenta", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="ink_cartridge_magenta_upright", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = 'z'
        self.z_offset = 0.1140
@register_object
class InkCartridgeMagentaFlat(IndustryObject):
    def __init__(
        self, 
        name="ink_cartridge_magenta_flat", 
        obj_name="ink_cartridge_magenta", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="ink_cartridge_magenta_flat", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.005

@register_object
class HeadsetInverted(IndustryObject):
    def __init__(
        self, 
        name="headset_inverted", 
        obj_name="headset", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="headset_inverted", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.005


@register_object
class MouseNagaUpright(IndustryObject):
    def __init__(
        self, 
        name="mouse_naga_upright", 
        obj_name="mouse_naga", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="mouse_naga_upright", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = 'z'
        self.z_offset = 0.0265
@register_object
class MouseNagaInverted(IndustryObject):
    def __init__(
        self, 
        name="mouse_naga_inverted", 
        obj_name="mouse_naga", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="mouse_naga_inverted", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.005
@register_object
class MouseWhiteTaipanUpright(IndustryObject):
    def __init__(
        self, 
        name="mouse_white_taipan_upright", 
        obj_name="mouse_white_taipan", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="mouse_white_taipan_upright", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = 'z'
        self.z_offset = 0.0207
@register_object
class MouseWhiteTaipanInverted(IndustryObject):
    def __init__(
        self, 
        name="mouse_white_taipan_inverted", 
        obj_name="mouse_white_taipan", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="mouse_white_taipan_inverted", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.005
@register_object
class MouseBlackTaipanUpright(IndustryObject):
    def __init__(
        self, 
        name="mouse_black_taipan_upright", 
        obj_name="mouse_black_taipan", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="mouse_black_taipan_upright", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = 'z'
        self.z_offset = 0.0200
@register_object
class MouseBlackTaipanInverted(IndustryObject):
    def __init__(
        self, 
        name="mouse_black_taipan_inverted", 
        obj_name="mouse_black_taipan", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="mouse_black_taipan_inverted", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.005
@register_object
class HeadsetUprightGroove(IndustryObject):
    def __init__(
        self,
        name="headset_upright_groove",
        obj_name="headset_upright_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class MouseBlackTaipanUprightGroove(IndustryObject):
    def __init__(
        self,
        name="mouse_black_taipan_upright_groove",
        obj_name="mouse_black_taipan_upright_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class MouseWhiteTaipanUprightGroove(IndustryObject):
    def __init__(
        self,
        name="mouse_white_taipan_upright_groove",
        obj_name="mouse_white_taipan_upright_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class MouseNagaUprightGroove(IndustryObject):
    def __init__(
        self,
        name="mouse_naga_upright_groove",
        obj_name="mouse_naga_upright_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class MouseBlackTaipanVGroove(IndustryObject):
    def __init__(
        self,
        name="mouse_black_taipan_v_groove",
        obj_name="mouse_black_taipan_v_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class MouseWhiteTaipanVGroove(IndustryObject):
    def __init__(
        self,
        name="mouse_white_taipan_v_groove",
        obj_name="mouse_white_taipan_v_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class MouseNagaVGroove(IndustryObject):
    def __init__(
        self,
        name="mouse_naga_v_groove",
        obj_name="mouse_naga_v_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class HeadsetFlatGroove(IndustryObject):
    def __init__(
        self,
        name="headset_flat_groove",
        obj_name="headset_flat_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class HeadsetVGroove(IndustryObject):
    def __init__(
        self,
        name="headset_v_groove",
        obj_name="headset_v_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100

@register_object
class ElectronicsTray(IndustryObject):
    def __init__(
        self, 
        name="electronics_tray", 
        obj_name="electronics_tray", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="electronics_tray", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = 'z'
        self.z_offset = 0.0100



@register_object
class MouseBlackTaipanUprightInlay(IndustryObject):
    def __init__(
        self,
        name="mouse_black_taipan_upright_inlay",
        obj_name="mouse_black_taipan_upright_inlay",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, xml_name="mouse_black_taipan_upright_inlay", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100

@register_object
class HeadsetUprightInlay(IndustryObject):
    def __init__(
        self,
        name="headset_upright_inlay",
        obj_name="headset_upright_inlay",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, xml_name="headset_upright_inlay", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100

@register_object
class MouseWhiteTaipanUprightInlay(IndustryObject):
    def __init__(
        self,
        name="mouse_white_taipan_upright_inlay",
        obj_name="mouse_white_taipan_upright_inlay",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, xml_name="mouse_white_taipan_upright_inlay", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100

@register_object
class MouseNagaUprightInlay(IndustryObject):
    def __init__(
        self,
        name="mouse_naga_upright_inlay",
        obj_name="mouse_naga_upright_inlay",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, xml_name="mouse_naga_upright_inlay", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class ThomasUpright(IndustryObject):
    def __init__(
        self, 
        name="thomas_upright", 
        obj_name="thomas", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="thomas_upright", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = 'z'
        self.z_offset = 0.0252
@register_object
class ThomasFlat(IndustryObject):
    def __init__(
        self, 
        name="thomas_flat", 
        obj_name="thomas", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="thomas_flat", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.005
@register_object
class PorterUpright(IndustryObject):
    def __init__(
        self, 
        name="porter_upright", 
        obj_name="porter", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="porter_upright", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = 'z'
        self.z_offset = 0.0250
@register_object
class PorterFlat(IndustryObject):
    def __init__(
        self, 
        name="porter_flat", 
        obj_name="porter", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="porter_flat", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.005
@register_object
class MarioUpright(IndustryObject):
    def __init__(
        self, 
        name="mario_upright", 
        obj_name="mario", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="mario_upright", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = 'z'
        self.z_offset = 0.0512
@register_object
class MarioFlat(IndustryObject):
    def __init__(
        self, 
        name="mario_flat", 
        obj_name="mario", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="mario_flat", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.005
@register_object
class MiniFireEngineUpright(IndustryObject):
    def __init__(
        self, 
        name="mini_fire_engine_upright", 
        obj_name="mini_fire_engine", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="mini_fire_engine_upright", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = 'z'
        self.z_offset = 0.0650
@register_object
class MiniFireEngineFlat(IndustryObject):
    def __init__(
        self, 
        name="mini_fire_engine_flat", 
        obj_name="mini_fire_engine", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="mini_fire_engine_flat", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.005
@register_object
class MiniRollerUpright(IndustryObject):
    def __init__(
        self, 
        name="mini_roller_upright", 
        obj_name="mini_roller", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="mini_roller_upright", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = 'z'
        self.z_offset = 0.0420
@register_object
class MiniRollerFlat(IndustryObject):
    def __init__(
        self, 
        name="mini_roller_flat", 
        obj_name="mini_roller", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="mini_roller_flat", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.005
@register_object
class SchoolBusUpright(IndustryObject):
    def __init__(
        self, 
        name="school_bus_upright", 
        obj_name="school_bus", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="school_bus_upright", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = 'z'
        self.z_offset = 0.0191
@register_object
class SchoolBusFlat(IndustryObject):
    def __init__(
        self, 
        name="school_bus_flat", 
        obj_name="school_bus", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="school_bus_flat", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.005
@register_object
class GrimlockUpright(IndustryObject):
    def __init__(
        self, 
        name="grimlock_upright", 
        obj_name="grimlock", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="grimlock_upright", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = 'z'
        self.z_offset = 0.0490
@register_object
class GrimlockFlat(IndustryObject):
    def __init__(
        self, 
        name="grimlock_flat", 
        obj_name="grimlock", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="grimlock_flat", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.005
@register_object
class DragonUpright(IndustryObject):
    def __init__(
        self, 
        name="dragon_upright", 
        obj_name="dragon", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="dragon_upright", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = 'z'
        self.z_offset = 0.0289
@register_object
class DragonFlat(IndustryObject):
    def __init__(
        self, 
        name="dragon_flat", 
        obj_name="dragon", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="dragon_flat", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.005
@register_object
class BumblebeeUpright(IndustryObject):
    def __init__(
        self, 
        name="bumblebee_upright", 
        obj_name="bumblebee", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="bumblebee_upright", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = 'z'
        self.z_offset = 0.0229
@register_object
class BumblebeeFlat(IndustryObject):
    def __init__(
        self, 
        name="bumblebee_flat", 
        obj_name="bumblebee", 
        joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(name, obj_name, xml_name="bumblebee_flat", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.005
@register_object
class ThomasVGroove(IndustryObject):
    def __init__(
        self,
        name="thomas_v_groove",
        obj_name="thomas_v_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class PorterVGroove(IndustryObject):
    def __init__(
        self,
        name="porter_v_groove",
        obj_name="porter_v_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class MarioVGroove(IndustryObject):
    def __init__(
        self,
        name="mario_v_groove",
        obj_name="mario_v_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class MiniFireEngineVGroove(IndustryObject):
    def __init__(
        self,
        name="mini_fire_engine_v_groove",
        obj_name="mini_fire_engine_v_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class MiniRollerVGroove(IndustryObject):
    def __init__(
        self,
        name="mini_roller_v_groove",
        obj_name="mini_roller_v_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class SchoolBusVGroove(IndustryObject):
    def __init__(
        self,
        name="school_bus_v_groove",
        obj_name="school_bus_v_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class ThomasFlatGroove(IndustryObject):
    def __init__(
        self,
        name="thomas_flat_groove",
        obj_name="thomas_flat_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class PorterFlatGroove(IndustryObject):
    def __init__(
        self,
        name="porter_flat_groove",
        obj_name="porter_flat_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class MarioFlatGroove(IndustryObject):
    def __init__(
        self,
        name="mario_flat_groove",
        obj_name="mario_flat_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0325

@register_object
class MiniFireEngineFlatGroove(IndustryObject):
    def __init__(
        self,
        name="mini_fire_engine_flat_groove",
        obj_name="mini_fire_engine_flat_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class MiniRollerFlatGroove(IndustryObject):
    def __init__(
        self,
        name="mini_roller_flat_groove",
        obj_name="mini_roller_flat_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class SchoolBusFlatGroove(IndustryObject):
    def __init__(
        self,
        name="school_bus_flat_groove",
        obj_name="school_bus_flat_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class ThomasUprightGroove(IndustryObject):
    def __init__(
        self,
        name="thomas_upright_groove",
        obj_name="thomas_upright_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class PorterUprightGroove(IndustryObject):
    def __init__(
        self,
        name="porter_upright_groove",
        obj_name="porter_upright_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class MarioUprightGroove(IndustryObject):
    def __init__(
        self,
        name="mario_upright_groove",
        obj_name="mario_upright_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class MiniFireEngineUprightGroove(IndustryObject):
    def __init__(
        self,
        name="mini_fire_engine_upright_groove",
        obj_name="mini_fire_engine_upright_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class MiniRollerUprightGroove(IndustryObject):
    def __init__(
        self,
        name="mini_roller_upright_groove",
        obj_name="mini_roller_upright_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100


@register_object
class SchoolBusUprightGroove(IndustryObject):
    def __init__(
        self,
        name="school_bus_upright_groove",
        obj_name="school_bus_upright_groove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100

@register_object
class MarioFlatInlay(IndustryObject):
    def __init__(
        self,
        name="mario_flat_inlay",
        obj_name="mario_flat_inlay",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, xml_name="mario_flat_inlay", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100

@register_object
class MiniFireEngineFlatInlay(IndustryObject):
    def __init__(
        self,
        name="mini_fire_engine_flat_inlay",
        obj_name="mini_fire_engine_flat_inlay",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, xml_name="mini_fire_engine_flat_inlay", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100

@register_object
class MiniRollerFlatInlay(IndustryObject):
    def __init__(
        self,
        name="mini_roller_flat_inlay",
        obj_name="mini_roller_flat_inlay",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, xml_name="mini_roller_flat_inlay", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100

@register_object
class PorterFlatInlay(IndustryObject):
    def __init__(
        self,
        name="porter_flat_inlay",
        obj_name="porter_flat_inlay",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, xml_name="porter_flat_inlay", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100

@register_object
class ThomasFlatInlay(IndustryObject):
    def __init__(
        self,
        name="thomas_flat_inlay",
        obj_name="thomas_flat_inlay",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, xml_name="thomas_flat_inlay", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100

@register_object
class SchoolBusFlatInlay(IndustryObject):
    def __init__(
        self,
        name="school_bus_flat_inlay",
        obj_name="school_bus_flat_inlay",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, xml_name="school_bus_flat_inlay", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100

@register_object
class AblationPlate(IndustryObject):
    def __init__(
        self,
        name="plate",
        obj_name="ablation_plate",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, xml_name="plate", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100

@register_object
class AblationAkitaBlackBowl(IndustryObject):
    def __init__(
        self,
        name="akita_black_bowl",
        obj_name="ablation_akita_black_bowl",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, xml_name="akita_black_bowl", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100

@register_object
class AblationCreamCheese(IndustryObject):
    def __init__(
        self,
        name="cream_cheese",
        obj_name="ablation_cream_cheese",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, xml_name="cream_cheese", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100

@register_object
class AblationTomatoSauce(IndustryObject):
    def __init__(
        self,
        name="tomato_sauce",
        obj_name="ablation_tomato_sauce",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, xml_name="tomato_sauce", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100

@register_object
class AblationBbqSauce(IndustryObject):
    def __init__(
        self,
        name="bbq_sauce",
        obj_name="ablation_bbq_sauce",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, xml_name="bbq_sauce", joints=joints)
        self.rotation = {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        self.rotation_axis = None
        self.z_offset = 0.0100
