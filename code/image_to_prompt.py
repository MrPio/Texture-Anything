from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# num_beams
# Specifica il numero di beam nel beam search; un valore >1 aumenta l’esplorazione di percorsi alternativi,
# migliorando la coerenza a scapito di maggior calcolo. Valori tipici vanno da 3 a 10
#
# do_sample
# Se False esegue beam search o greedy decoding, altrimenti attiva il campionamento multinomiale, favorendo creatività
# ma riducendo la coerenza. Spesso lo si imposta a True solo con parametri di sampling (vedi top_k/top_p)
#
# temperature
# Modula la “temperatura” della distribuzione di probabilità.
# <1.0: output più conservativo (meno variabilità).
# >1.0: output più creativo e variegato.
# Impostazioni tipiche tra 0.7 e 1.5

# top_k
# Limita il vocabolario ai K token più probabili ad ogni step, riducendo rumore e improbabilità.
# Solitamente si usa top_k=50 o più basso (es. 20) per maggior precisione
#
# top_p (nucleus sampling)
# Conserva il sottoinsieme minimo di token la cui probabilità cumulativa ≥top_p.
# top_p=0.9 favorisce coerenza con lieve creatività.
# top_p<0.5 molto conservativo.
# top_p≈1.0 equivale a campionamento puro
#
# 2. Controllo della lunghezza e ripetizioni: max_length / min_length
# max_length definisce il numero massimo di token prodotti, inclusi quelli del prompt.
# min_length impone una lunghezza minima prima di permettere la generazione di <eos>.
# Usali per evitare didascalie troppo brevi o eccessivamente estese (es. min_length=20, max_length=100)
#
# length_penalty
# Esponenziale penalità sulla lunghezza del beam search:
# >1.0 incoraggia sequenze più lunghe.
# <1.0 favorisce output più brevi.
# Imposta valori tra 0.8 e 1.2 a seconda del bilanciamento desiderato tra brevità e completezza

#
# repetition_penalty
# Penalizza la ripetizione di token già generati: >1.0 scoraggia la ripetizione.
# 1.0 nessuna penalità. Valori comuni intorno a 1.2–1.5 aiutano a eliminare loop ripetitivi nelle didascalie
#
# no_repeat_ngram_size
# Impedisce che n‑grammi di lunghezza n si ripetano.
# no_repeat_ngram_size=2/3 è efficace per evitare ripetizioni fastidiose mantenendo flessibilità
#
# early_stopping
# Con True, interrompe il beam search non appena ci sono num_beams candidati completi, riducendo il tempo di generazione a scapito di esplorazione aggiuntiva



gen_kwargs = {
    "num_beams": 10,            # beam search moderatoce
    "do_sample": False,         # sampling abilitato:c
    "temperature": 0.8,        # output più controllati
    "top_k": 20,               # filtro top-k per maggior precisione
    "top_p": 0.9,              # nucleus sampling bilanciato
    "length_penalty": 1.0,     # nessuna modifica alla lunghezza
    "repetition_penalty": 1.3, # penalizza ripetizioni
    "no_repeat_ngram_size": 2, # evita bigrammi ripetuti
    "max_length": 200,          # limita la lunghezza massima
    "min_length": 100,          # limita la lunghezza minima
    "early_stopping": True     # termina precocemente se possibile
}



def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds


print(predict_step(['pettirosso.png']))  # ['a woman in a hospital bed with a woman in a hospital bed']
