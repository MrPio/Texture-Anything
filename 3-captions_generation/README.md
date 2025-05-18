# Refine The Dataset

This section generates missing captions. To create captions, we need one or more renders of each object. Objaverse provides thumbnails, but ShapeNetCore needs a custom rendering script.

We test several vision transformers for captioning. In the end, we use [Microsoft's Phi 3.5](https://huggingface.co/microsoft/Phi-3.5-vision-instruct).