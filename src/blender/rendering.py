#!/usr/bin/env python3
"""
Renderizza una vista ortografica con camera fissa su un modello 3D che contiene già la texture.
"""

import argparse
import logging
from pathlib import Path
from PIL import Image
import numpy as np
import trimesh
import pyrender


def look_at(camera_pos, target, up=np.array([0, 1, 0])):
    """
    Calcola la matrice di rotazione per far 'guardare' un oggetto da camera_pos verso target.
    """
    forward = (target - camera_pos)
    forward /= np.linalg.norm(forward)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    true_up = np.cross(forward, right)

    rot = np.eye(4)
    rot[:3, 0] = right
    rot[:3, 1] = true_up
    rot[:3, 2] = forward
    return rot

def load_model(file_path: Path) -> trimesh.Trimesh:
    """
    Carica un modello 3D.
    """
    loaded = trimesh.load(file_path, force='scene')
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError("La scena non contiene geometrie.")
        mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ValueError(f"Tipo di mesh non supportato: {type(loaded)}")

    return mesh


def render(mesh: trimesh.Trimesh,
                       output_path: Path,
                       image_size: tuple[int, int] = (512, 512)):
    """
    Renderizza una vista ortografica con camera fissa.
    La texture deve essere già presente nel modello.
    """
    # Imposta la posizione fissa della camera
    position = np.array([-3.88422, 1.16845, 15.3794])
    rotation = np.eye(3)  # Nessuna rotazione
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = position

    # Calcola un ortho scale ragionevole in base alla bounding box
    bounds = mesh.bounds
    size = bounds[1] - bounds[0]
    xmag = size[0]
    ymag = size[1]

    # Crea camera ortografica
    camera = pyrender.OrthographicCamera(xmag=xmag, ymag=ymag)

    # Crea la scena
    scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[3, 3, 3])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    scene.add(camera, pose=pose)
    light_pose = look_at(position, target=np.array([0.0, 0.0, 0.0]))  # Verso l'origine
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=10), pose=light_pose)

    # Renderizza
    renderer = pyrender.OffscreenRenderer(*image_size)
    color, _ = renderer.render(scene)
    renderer.delete()

    # Salva l'immagine renderizzata
    out_img = Image.fromarray(color).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(output_path)
    logging.info(f"Immagine salvata in: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Renderizza una vista ortografica di un modello 3D con texture già presente.")
    parser.add_argument("input_file", nargs="?", type=Path, default=Path("oggetto.glb"),
                        help="Path al file 3D (default: oggetto.glb)")
    parser.add_argument("-o", "--output", type=Path, default=Path("rendering/rendering.png"),
                        help="Path del file immagine di output")
    parser.add_argument("-s", "--size", type=int, nargs=2, metavar=('W', 'H'),
                        default=[512, 512],
                        help="Dimensione immagine (larghezza altezza)")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    mesh = load_model(args.input_file)
    render(mesh, args.output, tuple(args.size))


if __name__ == "__main__":
    main()
