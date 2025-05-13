#!/usr/bin/env blender -b --python

import argparse
import sys
import logging
from pathlib import Path
import bpy
from mathutils import Vector

ADD_LAMP = True
DISTANCE = 1.0
BACKGROUND_COLOR = (1.0, 1.0, 1.0, 1.0)
SAMPLES = 1024

DEFAULT_INPUT_PATH = "prova.glb"   # <--- CAMBIA PER LE PROVE, ALTRIMENTI LANCIA DA CLI
DEFAULT_OUTPUT_PATH = "rendering/render"

# ——————————————————————————————————————————————————————————————————————————————
# UTILITIES
# ——————————————————————————————————————————————————————————————————————————————

def setup_logging() -> None:
    """Configura il logging di base."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )

def parse_args() -> argparse.Namespace:
    """Legge gli argomenti passati dopo --."""
    if '--' in sys.argv:
        idx = sys.argv.index('--')
        script_args = sys.argv[idx + 1:]
    else:
        script_args = []
    parser = argparse.ArgumentParser(
        description="Renderizza un modello 3D in modalità background con Blender")
    parser.add_argument(
        "-i" , "--input", type=Path,
        default=Path(DEFAULT_INPUT_PATH),
        help="Percorso al file 3D (.obj, .gltf, .glb)")
    parser.add_argument(
        "-o", "--output", type=Path,
        default=Path(DEFAULT_OUTPUT_PATH),
        help="Percorso del file PNG di output")
    parser.add_argument(
        "-s", "--size", type=int, nargs=2, metavar=('W','H'),
        default=[2048, 2048],
        help="Dimensione del render (width height)")
    parser.add_argument(
        "--transparent-bg", action='store_true',
        help="Attiva lo sfondo trasparente (film transparent)")
    return parser.parse_args(script_args)

def clear_scene() -> None:
    """Pulisce la scena caricando le impostazioni di fabbrica vuote."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    logging.info("Scena ripulita (factory settings).")

# ——————————————————————————————————————————————————————————————————————————————
# IMPORT OBJECT
# ——————————————————————————————————————————————————————————————————————————————
def load_object(filepath: Path) -> list[bpy.types.Object]:
    """Importa un modello 3D da file, centra tutto sull'origine e restituisce gli oggetti importati."""
    ext = filepath.suffix.lower()
    path_str = str(filepath.resolve())

    if ext == '.obj':
        bpy.ops.wm.obj_import(filepath=path_str)
    elif ext in ('.gltf', '.glb'):
        bpy.ops.import_scene.gltf(filepath=path_str)
    else:
        raise ValueError(f"Formato non supportato: '{ext}'")


    objs = bpy.context.selected_objects
    logging.info(f"Importati oggetti: {[o.name for o in objs]}")

    #Traslo l'oggetto al centro
    for o in objs:
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        o.location = (0.0, 0.0, 0.0)

    return objs

def setup_camera(target_objs):
    """
    Posiziona la camera in modo che inquadri il target_objs al centro della scena.
    La distanza viene calcolata come distanza_factor * dimensione massima del bounding box.
    """

    center = Vector((0.0, 0.0, 0.0))
    # Determina la dimensione massima del bounding box
    max_dim = 0.0
    for o in target_objs:
        if o.type == 'MESH' and o.data:
            dims = o.dimensions.length  # lunghezza diagonale di ciascun oggetto
            if dims > max_dim:
                max_dim = dims
    # Se non trovi nessuna mesh, usa una distanza di fallback
    if max_dim == 0.0:
        logging.warning("setup_camera: max_dim risulta zero, uso fallback distance = 1.0")
        max_dim = 1.0

    # Calcola la distanza in base alla dimensione
    distance = max_dim * DISTANCE

    # Crea e linka la camera
    cam_data = bpy.data.cameras.new('Camera')
    cam_obj  = bpy.data.objects.new('Camera', cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    # Posiziona la camera lungo la diagonale XYZ
    cam_obj.location = center + Vector((distance, distance, distance))

    # Orienta la camera verso il centro
    direction = center - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()


    # Debug
    print(f"[Camera] center: {center}")
    print(f"[Camera] max_dim: {max_dim}, factor: {DISTANCE}, distance: {distance}")
    print(f"[Camera] location: {cam_obj.location}, rotation: {cam_obj.rotation_euler}")

    return cam_obj

# ——————————————————————————————————————————————————————————————————————————————
# LIGHTING & WORLD
# ——————————————————————————————————————————————————————————————————————————————

def setup_lighting() -> None:
    """Aggiunge una sun light come key e una point light come fill."""
    # Key light
    key_data = bpy.data.lights.new('KeyLight', 'SUN')
    key = bpy.data.objects.new('KeyLight', key_data)
    bpy.context.collection.objects.link(key)
    key.rotation_euler = (0.7854, 0, 0.7854)

    # Fill light
    fill_data = bpy.data.lights.new('FillLight', 'POINT')
    fill = bpy.data.objects.new('FillLight', fill_data)
    bpy.context.collection.objects.link(fill)

    if ADD_LAMP :
        # Posiziono a metà strada camera–centro
        cam_loc = bpy.context.scene.camera.location
        fill.location = cam_loc * 0.5
        key_data.energy = 1.0
        fill_data.energy = 30.0
    logging.info("Lighting setup completo.")

def configure_world(transparent: bool=False) -> None:
    """Imposta uno sfondo neutro o trasparente."""
    scene = bpy.context.scene
    world = bpy.data.worlds.new("RenderWorld")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    if transparent:
        scene.render.film_transparent = True
        logging.info("Sfondo impostato trasparente.")
    else:
        bg.inputs[0].default_value = BACKGROUND_COLOR

# ——————————————————————————————————————————————————————————————————————————————
# RENDER
# ——————————————————————————————————————————————————————————————————————————————

def configure_render(output: Path, input:Path, size: tuple[int,int]) -> None:
    """Configura engine, risoluzione e percorso di output."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = SAMPLES
    scene.render.resolution_x, scene.render.resolution_y = size
    scene.render.filepath = str(output.resolve())+ "_"+str(input)[:-4]+".png"
    scene.render.image_settings.file_format = 'PNG'
    logging.info(f"Render configurato: {size[0]}×{size[1]}, output: {scene.render.filepath}")

def render_scene() -> None:
    """Esegue il render e salva l’immagine."""
    bpy.ops.render.render(write_still=True)
    logging.info("Render completo.")

# ——————————————————————————————————————————————————————————————————————————————
# MAIN
# ——————————————————————————————————————————————————————————————————————————————

def main() -> None:
    setup_logging()
    args = parse_args()
    clear_scene()
    objs = load_object(args.input)
    setup_camera(objs)
    configure_world(args.transparent_bg)
    setup_lighting()
    configure_render(args.output, args.input, tuple(args.size))
    render_scene()

if __name__ == '__main__':
    main()
