# Introducing the ShapeNetCore dataset

In this section, we download and preprocess the _ShapeNetCore_ dataset. The process is similar to sections 1 and 2 for Objaverse, but ShapeNetCore has no annotations, so we download the entire dataset.

ShapeNetCore objects are more consistent in size and style, but the UV maps are often poor. Without regenerating UV maps and baking textures ([`4.1-Regenerate_UV_Maps.ipynb`](../4-enhance_dataset/4.1-Regenerate_UV_Maps.ipynb) shows an attempt), this dataset is not used further.

> [!Important]Execution Order  
> First download the objects using [`3.1-Download_ShapeNet.ipynb`](3.1-Download_ShapeNet.ipynb). Then run [`generate_statistics.py`](../2-objects_filtering/generate_statistics.py) and [`generate_dataset.py`](../2-objects_filtering/generate_dataset.py) using `ShapeNetCoreDataset3D` instead of `ObjaverseDataset3D`.
