# Objaverse Objects Filtering

First, we download the ~80,000 object UIDs selected earlier.

Next, we compute statistics for each object: `meshCount`, `uvCount`, `diffuseCount`, and `uvScore`. Using Blender (`bpy`), we select only objects with 1 mesh, 1 UV, 1 diffuse texture, and a good UV layout. In total, we end up with ~14,000 samples.

> [!Note]UV Score  
> UV quality is measured by how well the 3D face shapes are preserved in the 2D UV space.

Finally, we create the dataset as triplets:

* **Render:** Thumbnail of the object (used for captioning in the next step)
* **UV:** UV map drawing
* **Diffuse:** Diffuse texture (used as ground truth for ControlNet training)

> [!Important]Execution Order  
> First download the objects using [`2.1-Download_Objaverse.ipynb`](2.1-Download_Objaverse.ipynb). Then run [`generate_statistics.py`](generate_statistics.py) and [`generate_dataset.py`](generate_dataset.py). All these 3 steps require a good amount of time.
