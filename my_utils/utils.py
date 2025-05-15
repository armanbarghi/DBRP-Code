import os
import numpy as np
import trimesh
import pybullet as p

def decompose_obj(directory, obj_name):
    input_path = os.path.join(directory, f"{obj_name}.obj")
    output_path = os.path.join(directory, f"{obj_name}_vhacd.obj")
    log_path = os.path.join(directory, "vhacd_log.txt")

    p.vhacd(
        input_path,
        output_path,
        log_path,
        resolution=100000,
        depth=20,
        concavity=0.0025,
        planeDownsampling=4,
        convexhullDownsampling=4,
        alpha=0.04,
        beta=0.05,
        gamma=0.00125,
        minVolumePerCH=0.0001
    )

    print(f"[VHACD] Decomposition complete: {output_path}")

def create_urdf(directory, name, mass, concave=False):
    obj_path = os.path.join(directory, f"{name}.obj")
    scene = trimesh.load(obj_path, force='scene')
    parts = []

    if len(scene.geometry) == 1:
        parts.append(("link_1", f"{name}.obj"))
    else:
        for idx, (geom_name, geom) in enumerate(scene.geometry.items(), start=1):
            part_filename = f"{name}_{idx}.obj"
            part_path = os.path.join(directory, part_filename)

            if hasattr(geom.visual, 'material') and geom.visual.material is not None:
                geom.visual.material.name = geom_name
            else:
                geom.visual.material = trimesh.visual.material.SimpleMaterial(name=geom_name)

            obj_text = trimesh.exchange.obj.export_obj(geom, mtl_name='material.mtl')

            lines = obj_text.splitlines()
            for i, line in enumerate(lines):
                if line.startswith('mtllib'):
                    lines.insert(i + 1, f'o {name}_{idx}')
                    break
            obj_text_with_o = '\n'.join(lines)

            with open(part_path, 'w') as f:
                f.write(obj_text_with_o)

            parts.append((f"link_{idx}", part_filename))

    collision_file = f"{name}_vhacd.obj" if concave else f"{name}.obj"
    if concave:
        decompose_obj(directory, name)

    links_xml = []
    joints_xml = []
    for i, (link_name, mesh_file) in enumerate(parts):
        links_xml.append(f'''
  <link name="{link_name}">
    <inertial>
      <mass value="{mass / len(parts):.4f}"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
    <visual>
      <geometry><mesh filename="{mesh_file}"/></geometry>
    </visual>
    <collision>
      <geometry><mesh filename="{collision_file}"/></geometry>
    </collision>
  </link>''')

        if i > 0:
            joints_xml.append(f'''
  <joint name="fixed_{parts[i - 1][0]}_{link_name}" type="fixed">
    <parent link="{parts[i - 1][0]}"/>
    <child link="{link_name}"/>
  </joint>''')

    urdf_content = f'''<?xml version="1.0"?>
<robot name="{name}">
{''.join(links_xml)}
{''.join(joints_xml)}
</robot>'''

    urdf_path = os.path.join(directory, f"{name}.urdf")
    with open(urdf_path, 'w') as f:
        f.write(urdf_content)

    print(f"[URDF] Created with {len(parts)} link(s): {urdf_path}")

def create_obj(obj_name, mass, concave=False):
    base_path = os.path.join("objects", obj_name)

    for subdir in os.listdir(base_path):
        folder_path = os.path.join(base_path, subdir)

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and not (filename.endswith('.glb') or filename.endswith('.blend')):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"[Cleanup] Error deleting {file_path}: {e}")

        glb_path = os.path.join(folder_path, f"{subdir}.glb")
        obj_path = os.path.join(folder_path, f"{subdir}.obj")

        scene = trimesh.load(glb_path, force='scene')
        scene.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        scene.export(obj_path)
        print(f"[Export] Converted to OBJ: {obj_path}")

        create_urdf(folder_path, subdir, mass, concave)
