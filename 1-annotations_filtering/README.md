# Objaverse Annotations Filtering

>[!IMPORTANT]Execution Order
>The notebooks in this section form a pipeline and must be executed in order.

We use _Objaverse 1.0_ (~800,000 objects) instead of Objaverse XL (~8,000,000 objects) because its metadata is cleaner.

We retrieve the annotations, remove unnecessary columns, and select one thumbnail per sample.

Finally, we filter the dataset by face count, reducing it to ~80,000 objects.
