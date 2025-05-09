__author__ = "Valerio Morelli, Mattia Sbattella, Jacopo Coloccioni"
__email__ = "valeriomorelli50@gmail.com"
__license__ = "Apache-2.0"

from .blender.scene import load_model, get_scene_stats
from .blender.object3d.object3d import Object3D # TODO comment
from .blender.object3d.shapnetcore_object3d import ShapNetCoreObject3D
from .blender.object3d.objaverse_object3d import ObjaverseObject3D
from .shapenet.dataset import load_shapenetcore_objects, load_annotations
from .utils import plot_images, compute_opacity
import logging

# Disable BPY annoying logging on stdout
logging.disable(logging.CRITICAL)
