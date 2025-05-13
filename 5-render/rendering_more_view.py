#!/usr/bin/env blender -b --python

import argparse
import sys
import logging
from pathlib import Path
import bpy
from mathutils import Vector
import math

ADD_LAMP = True
DISTANCE = 1.5
BACKGROUND_COLOR = (1.0, 1.0, 1.0, 1.0)
SAMPLES = 256
QUALITY = 512
DEFAULT_INPUT_PATH = "prova.glb"   # <--- CAMBIA PER LE PROVE, ALTRIMENTI LANCIA DA CLI
DEFAULT_OUTPUT_PATH = "rendering/render"
# ——————————————————————————————————————————————————————————————————————————————
# UTILITIES
# ——————————————————————————————————————————————————————————————————————————————

def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def parse_args() -> argparse.Namespace:
    if '--' in sys.argv:
        idx = sys.argv.index('--')
        script_args = sys.argv[idx + 1:]
    else:
        script_args = []
    p = argparse.ArgumentParser(description="Render 4 viste intorno all'oggetto")
    p.add_argument("-i","--input", type=Path, default=Path(DEFAULT_INPUT_PATH))
    p.add_argument("-o","--output", type=Path, default=Path(DEFAULT_OUTPUT_PATH))
    p.add_argument("-s","--size", type=int, nargs=2, default=[QUALITY,QUALITY])
    p.add_argument("--transparent-bg", action='store_true')
    return p.parse_args(script_args)

def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    logging.info("Scena ripulita")

# ——————————————————————————————————————————————————————————————————————————————
# IMPORT & CENTER
# ——————————————————————————————————————————————————————————————————————————————

def load_object(filepath: Path) -> list[bpy.types.Object]:
    ext = filepath.suffix.lower()
    path_str = str(filepath.resolve())
    if ext == '.obj':
        bpy.ops.wm.obj_import(filepath=path_str)
    elif ext in ('.gltf', '.glb'):
        bpy.ops.import_scene.gltf(filepath=path_str)
    else:
        raise ValueError(f"Formato non supportato: '{ext}'")
    objs = bpy.context.selected_objects
    logging.info(f"Importati: {[o.name for o in objs]}")
    # Centra ogni mesh all'origine
    for o in objs:
        if o.type == 'MESH':
            bpy.context.view_layer.objects.active = o
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
            o.location = (0,0,0)
    return objs

# ——————————————————————————————————————————————————————————————————————————————
# CAMERA SETUP (4 VIEWS)
# ——————————————————————————————————————————————————————————————————————————————

def setup_camera(target_objs):
    """
    Restituisce 4 viste intorno al centro:
     - angoli 0°, 90°, 180°, 270° sul piano XY
    """
    # calcola max_dim
    max_dim = 0.0
    for o in target_objs:
        if o.type=='MESH' and o.data:
            max_dim = max(max_dim, o.dimensions.length)
    if max_dim == 0.0:
        max_dim = 1.0
    distance = max_dim * DISTANCE

    # crea camera
    cam_data = bpy.data.cameras.new('Camera')
    cam_obj  = bpy.data.objects.new('Camera', cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    views = []
    # 4 angoli
    for i in range(4):
        ang = math.radians(45 + (90 * i))
        loc = Vector((distance * math.cos(ang),
                      distance * math.sin(ang),
                      0.0))
        quat = (Vector((0,0,0)) - loc).to_track_quat('-Z','Y')
        rot = quat.to_euler()
        views.append((loc, rot))
    return views

# ——————————————————————————————————————————————————————————————————————————————
# LIGHTING & WORLD
# ——————————————————————————————————————————————————————————————————————————————

def setup_lighting() -> None:
    key_data = bpy.data.lights.new('KeyLight','SUN')
    key = bpy.data.objects.new('KeyLight', key_data)
    bpy.context.collection.objects.link(key)
    key.rotation_euler = (0.7854,0,0.7854)
    fill_data = bpy.data.lights.new('FillLight','POINT')
    fill = bpy.data.objects.new('FillLight', fill_data)
    bpy.context.collection.objects.link(fill)
    if ADD_LAMP:
        cam_loc = bpy.context.scene.camera.location
        fill.location = cam_loc * 0.5
        key_data.energy = 1.0
        fill_data.energy = 30.0
    logging.info("Luci impostate")

def configure_world(transparent: bool=False) -> None:
    scene = bpy.context.scene
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    if transparent:
        scene.render.film_transparent = True
    else:
        bg.inputs[0].default_value = BACKGROUND_COLOR

# ——————————————————————————————————————————————————————————————————————————————
# RENDER
# ——————————————————————————————————————————————————————————————————————————————

def configure_render(size: tuple[int,int]) -> None:
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = SAMPLES
    scene.render.resolution_x, scene.render.resolution_y = size

def render_main(views, output: Path, input: Path) -> None:
    scene = bpy.context.scene
    base = str(output.resolve())+ "_"+str(input)[:-4]
    for idx, (loc, rot) in enumerate(views):
        cam = scene.camera
        cam.location = loc
        cam.rotation_euler = rot
        scene.render.filepath = base + f"_{idx:02d}" + ".png"
        bpy.ops.render.render(write_still=True)
        logging.info(f"Saved {scene.render.filepath}")

# ——————————————————————————————————————————————————————————————————————————————
# MAIN
# ——————————————————————————————————————————————————————————————————————————————

def main() -> None:
    setup_logging()
    args = parse_args()
    clear_scene()
    objs = load_object(args.input)
    views = setup_camera(objs)
    configure_world(args.transparent_bg)
    setup_lighting()
    configure_render(tuple(args.size))
    render_main(views, args.output, args.input)

if __name__ == '__main__':
    main()





