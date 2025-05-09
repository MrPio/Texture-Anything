__author__ = "Valerio Morelli, Mattia Sbattella, Jacopo Coloccioni"
__email__ = "valeriomorelli50@gmail.com"
__license__ = "Apache-2.0"

from .blender.scene import load_glb, get_scene_stats
from .blender.object import get_diffuse_textures, get_mesh_stats, draw_uv_map
from .shapenet.dataset import load_shapenetcore_objects, load_annotations
from .utils import plot_images, compute_opacity

import logging

logging.disable(logging.CRITICAL)
