# Dataset Creation

>[!Note]The Dataset Is Modular.
>You can add your own dataset by adding your own implementation of [`Dataset3D`](../src/dataset/dataset3d.py) and of [`Object3D`](../src/blender/object3d/object3d.py), if you want to work with the 3d models to generate the triplets.

First, we download the ~80,000 object UIDs selected earlier for Objaverse and the entire ShapeNetCore dataset of ~71,000 objects.

Next, we compute statistics for each object: `meshCount`, `uvCount`, `diffuseCount`, and `uvScore`. Using Blender (`bpy`), we select only objects with 1 mesh, 1 UV, 1 diffuse texture, and a good UV layout. In total, we end up with a total of ~14,000 samples for objaverse and a few thousands for SapeNetCore.

ShapeNetCore objects are more consistent in size and style, but the UV maps are often poor.

> [!Note]UV Score  
> UV quality is measured by how well the 3D face shapes are preserved in the 2D UV space.

Finally, we create the dataset as triplets:

* **Render:** Thumbnail of the object (used for captioning in the next step)
* **UV:** UV map drawing
* **Diffuse:** Diffuse texture (used as ground truth for ControlNet training)

> [!Note]Execution Time  
> Downloading the Objaverse dataset may take some time because we are interested in a subset of the dataset and ~80,000 HTTP requests are required. If we parallelize too much, we will be temporarily blocked by HuggingFace.

> [!Important]Execution Order  
> First download the objects using `2.1`. Then run [`generate_statistics.py`](generate_statistics.py) and [`generate_dataset.py`](generate_dataset.py) first with the argument `--dataset="objaverse"`, and then with `--dataset="shapenetcore"`.