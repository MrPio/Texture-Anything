# KANBAN

## TODO

    - [ ] Data Augmentation:se abbiamo un certo insieme di oggetti con UV particolarmente buona e texture generica, possiamo moltiplicare i samples scambiando le texture
    - [-] Rigenerare le texture/UV di Objaverse con smart
    - [-] La MSE non deve pesare le zone non mappate: EDIT: non è possibile farlo. Però possiamo rndere nere le zone non mappate

## DONE

    - [x] In infer_controlnet, usa il test set generato in 4.1, in modo da non usare elementi del trainset
    - [x] Provare 2.1 con dreambooth texture hell
    - [x] Fare script di inferenza di CNET
    - [x] Invertire le UV
    - [x] Add 3.1 in download method of ShapeNetCoreDataset3D
    - [x] Generate the rendering of objects
    - [x] Define a UV map scoring function
    - [x] Generate captions from renders
    - [x] Objects statistics have been increased to 45_000. Launch generate_dataset.py.
    - [x] Try shapenetcore, now they've granted me access
    - [x] For some objects, the smallest thumbnail is kept. (E.g. "91ed3c6a3cce40b5adead6f1e433a803")
      - Solved by pruning small thumbnails
