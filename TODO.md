# KANBAN

## TODO

    - [ ] The loss should not care about part of the texture not mapped?
    - [ ] Generate captions from renders
    - [ ] Generate the rendering of objects
    - [ ] _Data Augmentation_:se abbiamo un certo insieme di oggetti con UV particolarmente buona e texture generica, possiamo moltiplicare i samples scambiando le texture
    - [ ] Define a UV map scoring function

## DONE

    - [x] Objects statistics have been increased to 45_000. Launch generate_dataset.py.
    - [x] Try shapenetcore, now they've granted me access
    - [x] For some objects, the smallest thumbnail is kept. (E.g. "91ed3c6a3cce40b5adead6f1e433a803")
      - Solved by pruning small thumbnails
