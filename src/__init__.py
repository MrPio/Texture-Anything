__author__ = "Valerio Morelli, Mattia Sbattella, Jacopo Coloccioni"
__email__ = "valeriomorelli50@gmail.com"
__license__ = "Apache-2.0"

# Blender
from .blender.scene import load_model
from .blender.object3d.shapenetcore_object3d import ShapeNetCoreObject3D
from .blender.object3d.objaverse_object3d import ObjaverseObject3D

# Datasets
from .dataset.objaverse_dataset3d import ObjaverseDataset3D, OBJAVERSE_PATH
from .dataset.shapenetcore_dataset3d import ShapeNetCoreDataset3D

# Misc
from .utils import plot_images, compute_image_density

# Disable BPY annoying logging on stdout
import logging
logging.disable(logging.CRITICAL)

import objaverse
objaverse._VERSIONED_PATH = str(OBJAVERSE_PATH)


import seaborn as sns
sns.set(style="darkgrid", context="notebook", font_scale=1.15)
