__author__ = "Valerio Morelli, Mattia Sbattella, Jacopo Coloccioni"
__email__ = "valeriomorelli50@gmail.com"
__license__ = "Apache-2.0"


# Blender
from .blender.scene import load_model, load_hdri
from .blender.object3d.object3d import Object3D
from .blender.object3d.objaverse_object3d import ObjaverseObject3D
from .blender.object3d.shapenetcore_object3d import ShapeNetCoreObject3D

# Datasets
from .dataset.dataset3d import Dataset3D
from .dataset.objaverse_dataset3d import ObjaverseDataset3D
from .dataset.shapenetcore_dataset3d import ShapeNetCoreDataset3D

datasets: dict[str, type[Dataset3D]] = {
    "objaverse": ObjaverseDataset3D,
    "shapenetcore": ShapeNetCoreDataset3D,
}

# Filters
from .filter import SobelFilter, PrewittFilter, LaplacianFilter

# Misc
from .utils import plot_images, compute_image_density, is_textured
from .logger import log, cprint

# Initialization
import seaborn as sns

sns.set(style="darkgrid", context="notebook", font_scale=1.15)
