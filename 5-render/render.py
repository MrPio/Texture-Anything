import bpy
import os
import math
from mathutils import Vector

# === Percorsi ===
OBJ_PATH = r"7dd57384aa290a835821cea205f2a4bb\models\model_normalized.obj"
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "renders")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Pulizia scena ===
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)

# === Importazione OBJ ===
def import_obj(filepath):
    bpy.ops.wm.obj_import(filepath=filepath)

# === Setup luce ===
def setup_light():
    light_data = bpy.data.lights.new(name="Light", type='SUN')
    light = bpy.data.objects.new(name="Light", object_data=light_data)
    bpy.context.collection.objects.link(light)
    light.location = (10, -10, 10)
    light.data.energy = 10

# === Setup mondo bianco ===
def setup_world():
    bpy.context.scene.world.use_nodes = True
    bg = bpy.context.scene.world.node_tree.nodes['Background']
    bg.inputs[0].default_value = (1, 1, 1, 1)  # Bianco

# === Setup camera ===
def setup_camera():
    cam_data = bpy.data.cameras.new(name="Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    return cam

# === Centra oggetto e calcola bounding box ===
def center_object(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0, 0, 0)

    bpy.context.view_layer.update()
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = sum(bbox, Vector()) / 8.0
    max_radius = max((v - center).length for v in bbox)
    return max_radius

# === Posiziona camera ruotandola attorno all'oggetto ===
def position_camera(cam, radius, angle_deg):
    angle_rad = math.radians(angle_deg)
    x = radius * math.cos(angle_rad)
    y = radius * math.sin(angle_rad)
    z = radius * 0.5  # elevazione

    cam.location = (x, y, z)
    cam.data.type = 'PERSP'
    cam.data.lens = 35
    cam.data.clip_end = 1000
    cam.data.clip_start = 0.1

    direction = Vector((0, 0, 0)) - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()

# === Renderizza e salva ===
def render_view(filepath):
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.filepath = filepath
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.ops.render.render(write_still=True)

# === Main ===
def main():
    clear_scene()
    setup_world()
    import_obj(OBJ_PATH)
    setup_light()
    cam = setup_camera()

    obj = [o for o in bpy.context.scene.objects if o.type == 'MESH'][0]
    max_radius = center_object(obj)

    distance = max_radius * 3

    angles = [0, 90, 180, 270]
    names = ['front', 'right', 'back', 'left']

    for angle, name in zip(angles, names):
        position_camera(cam, distance, angle)
        output_path = os.path.join(OUTPUT_DIR, f"{name}.png")
        print(f"Rendering {name} to {output_path}")
        render_view(output_path)

if __name__ == "__main__":
    main()
