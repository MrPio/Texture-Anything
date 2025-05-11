__author__ = "Valerio Morelli, Mattia Sbattella, Jacopo Coloccioni"
__email__ = "valeriomorelli50@gmail.com"
__license__ = "Apache-2.0"

from .blender.scene import load_model

# from .blender.object3d.object3d import Object3D
from .blender.object3d.shapenetcore_object3d import ShapeNetCoreObject3D
from .blender.object3d.objaverse_object3d import ObjaverseObject3D
from .dataset.objaverse_dataset3d import ObjaverseDataset3D
from .dataset.shapenetcore_dataset3d import ShapeNetCoreDataset3D
from .utils import plot_images, compute_image_density
import logging

# Disable BPY annoying logging on stdout
logging.disable(logging.CRITICAL)
