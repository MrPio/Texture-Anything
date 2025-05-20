# KANBAN

## TODO

    - [ ] La MSE non deve pesare le zone non mappate
    - [ ] Data Augmentation:se abbiamo un certo insieme di oggetti con UV particolarmente buona e texture generica, possiamo moltiplicare i samples scambiando le texture
    - [ ] Fare script di inferenza di CNET
    - [ ] Provare 2.1 con dreambooth texture hell
    - [ ] Rigenerare le texture/UV di Objaverse con smart
    - [ ] Invertire le UV

## DONE

    - [x] Add 3.1 in download method of ShapeNetCoreDataset3D
    - [x] Generate the rendering of objects
    - [x] Define a UV map scoring function
    - [x] Generate captions from renders
    - [x] Objects statistics have been increased to 45_000. Launch generate_dataset.py.
    - [x] Try shapenetcore, now they've granted me access
    - [x] For some objects, the smallest thumbnail is kept. (E.g. "91ed3c6a3cce40b5adead6f1e433a803")
      - Solved by pruning small thumbnails
