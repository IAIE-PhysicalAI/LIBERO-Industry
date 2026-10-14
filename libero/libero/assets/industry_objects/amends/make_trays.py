import textwrap

# ==========================================
# 📝 VARIABLES TO FILL IN
# ==========================================
TRAY_NAME = "grey_tray"
LENGTH    = 0.25   # Total length along X-axis (meters)
WIDTH     = 0.25  # Total width along Y-axis (meters)
HEIGHT    = 0.06   # Total height of the tray walls (meters)
THICKNESS = 0.005   # Wall and base thickness (meters)
MASS      = 0.8    # Total mass of the tray

# 🆕 Grid Parameters for Placement
GRID_COLS = 3
GRID_ROWS = 2
MARGIN    = 0.02  # 2.5cm clearance from the walls for gripper handling

# ==========================================
# ⚙️ XML GENERATION LOGIC
# ==========================================
def generate_strict_tray_xml(name, length, width, height, thickness, mass):
    # MuJoCo uses half-extents for 'size'
    hx = length / 2.0
    hy = width / 2.0
    ht = thickness / 2.0
    
    # Wall height above the base
    wall_h = height - thickness
    hz = wall_h / 2.0
    z_center = thickness + hz
    
    # 1. Base Geometry
    base_size = f"{hx:.3f} {hy:.3f} {ht:.3f}"
    base_pos = f"0 0 {ht:.3f}"
    
    # 2. Left / Right Walls (Span the full Y width)
    lr_size = f"{ht:.3f} {hy:.3f} {hz:.3f}"
    l_pos = f"{-hx + ht:.3f} 0 {z_center:.3f}"
    r_pos = f"{hx - ht:.3f} 0 {z_center:.3f}"
    
    # 3. Front / Back Walls (Fit exactly between Left/Right walls)
    fb_hx = hx - thickness
    fb_size = f"{fb_hx:.3f} {ht:.3f} {hz:.3f}"
    f_pos = f"0 {-hy + ht:.3f} {z_center:.3f}"
    b_pos = f"0 {hy - ht:.3f} {z_center:.3f}"
    
    # 4. Master Goal Region Site (Covers the whole inside)
    site_size = f"{fb_hx:.3f} {hy - thickness:.3f} 0.001"
    site_pos = f"0 0 {thickness + 0.005:.3f}"
    
    # =========================================================
    # 5. 🆕 6 NON-OVERLAPPING PLACEMENT REGIONS
    # =========================================================
    avail_x = (fb_hx - MARGIN) * 2.0
    avail_y = ((hy - thickness) - MARGIN) * 2.0
    
    step_x = avail_x / GRID_COLS
    step_y = avail_y / GRID_ROWS
    
    # Size of the sub-regions (with a 2mm internal gap between them)
    sub_hx = (step_x / 2.0) - 0.01
    sub_hy = (step_y / 2.0) - 0.01
    
    start_x = -(avail_x / 2.0) + (step_x / 2.0)
    start_y = -(avail_y / 2.0) + (step_y / 2.0)
    
    regions_xml = ""
    region_idx = 1
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cx = start_x + c * step_x
            cy = start_y + r * step_y
            
            regions_xml += f'                  <site name="place_region_{region_idx}" type="box" size="{sub_hx:.3f} {sub_hy:.3f} 0.001" pos="{cx:.3f} {cy:.3f} {thickness + 0.005:.3f}" quat="1 0 0 0" rgba="0 0 1 0.2" group="0" />\n'
            region_idx += 1
            
    # =========================================================

    # 6. Robosuite Boundary Math
    bottom_z = -0.01  
    top_z = THICKNESS
    
    # STRICT HIERARCHY XML
    xml = f"""\
    <mujoco model="{name}">
      <worldbody>
        <body>
          <body name="object" pos="0 0 0" quat="1 0 0 0"> 
              <body name="{name}" pos="0 0 0" quat="1 0 0 0"> 
                  <inertial pos="0 0 0" mass="{mass}" diaginertia="1e-3 1e-3 1e-3" />
                  
                  <geom name="bottom" type="box" size="{base_size}" pos="{base_pos}" rgba="0.5 0.5 0.5 1" friction="0.8 0.005 0.001" group="1" conaffinity="1" contype="1"/>
                  
                  <geom name="wall_left" type="box" size="{lr_size}" pos="{l_pos}" rgba="0.5 0.5 0.5 1" friction="0.6 0.005 0.001" group="1" conaffinity="1" contype="1"/>
                  <geom name="wall_right" type="box" size="{lr_size}" pos="{r_pos}" rgba="0.5 0.5 0.5 1" friction="0.6 0.005 0.001" group="1" conaffinity="1" contype="1"/>
                  p
                  <geom name="wall_front" type="box" size="{fb_size}" pos="{f_pos}" rgba="0.5 0.5 0.5 1" friction="0.6 0.005 0.001" group="1" conaffinity="1" contype="1"/>
                  <geom name="wall_back" type="box" size="{fb_size}" pos="{b_pos}" rgba="0.5 0.5 0.5 1" friction="0.6 0.005 0.001" group="1" conaffinity="1" contype="1"/>

                  <site name="goal_region" type="box" size="{site_size}" pos="{site_pos}" quat="1 0 0 0" rgba="0 1 0 0.1" group="0" />
{regions_xml}
              </body>
          </body>
          
          <site rgba="0 0 0 0" size="0.01" pos="0 0 {bottom_z:.3f}" quat="1 0 0 0" name="bottom_site"/>        
          <site rgba="0 0 0 0" size="0.01" pos="0 0 {top_z:.3f}" quat="1 0 0 0" name="top_site"/>   
          <site rgba="0 0 0 0" size="0.01" pos="{hx:.3f} {hy:.3f} 0" quat="1 0 0 0" name="horizontal_radius_site"/>
        </body>
      </worldbody>
    </mujoco>"""
    
    return textwrap.dedent(xml)

# ==========================================
# 🖨️ PRINT OUTPUT
# ==========================================
if __name__ == "__main__":
    xml_output = generate_strict_tray_xml(TRAY_NAME, LENGTH, WIDTH, HEIGHT, THICKNESS, MASS)
    print("\n")
    print(xml_output)