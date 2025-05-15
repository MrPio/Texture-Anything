# Refine The Dataset

>[!Note]The Dataset Is Modular.
>You can add your own dataset by adding your own implementation of [`Dataset3D`](../src/dataset/dataset3d.py) and of [`Object3D`](../src/blender/object3d/object3d.py), if you want to work with the 3d models to generate the triplets.

This section generates missing captions and explores ways to improve the dataset. To create captions, we need one or more renders of each object. Objaverse provides thumbnails, but ShapeNetCore needs a custom rendering script.

We test several vision transformers for captioning. In the end, we use [Microsoft's Phi 3.5](https://huggingface.co/microsoft/Phi-3.5-vision-instruct).

Finally, we create a scoring function to measure how well the UV maps match the 3D object's shape.
