# Named Entity Recognition avec PyTorch

https://github.com/dreamgonfly/BERT-pytorch/blob/master/bert/train/model/transformer.py

Ce repository contient des tutoriels sur comment faire de l'analyse de sentiments en utilisant Pytorch 1.4 sur Python 3.7. 

## Installation 

 - Pour installer PyTorch, les instructions sont sur [ce site.](https://pytorch.org/get-started/locally/) 

 - TorchText : ` pip install torchtext`
 
 - spaCy en anglais : `python -m spacy download en`
 
 - transformers : `pip install transformers`
## Données

On utilise les données [CONLL2003](https://github.com/davidsbatista/NER-datasets/tree/master/CONLL2003)
## Tutoriels

 - 1 - [BiLSTM](https://github.com/aminaghoul/NER-PyTorch/blob/master/1-BiLSTM-NER.ipynb) 
 Dans ce premier notebook, le modèle utilisé est un LSTM bidirectionnel classique pour faire de la reconnaissance d'entités nommées (NER).

 - 2 - [BiLSTM+Embedding de caractère](https://github.com/aminaghoul/NER-PyTorch/blob/master/2-CharacterEmbedding-NER.ipynb]
 
Ici, on ajoute au modèle précédent un embedding de caractères en utilisant des convolutions.

 - 3 - [BiLSTM+CRF](https://github.com/aminaghoul/NER-PyTorch/blob/master/3-CRF-NER.ipynb)
 
 On ajoute au premier modèle (BiLSTM) une couche CRF.
 
 - 4 - [Attention](https://github.com/aminaghoul/NER-PyTorch/blob/master/4-Attention-NER-CONLL.ipynb) 
 
Dans ce notebook, on ajoute au modèle précédent une couche Attention. 

- 5 - [Transformers](https://github.com/aminaghoul/NER-PyTorch/blob/master/4-Attention-NER-CONLL.ipynb) 
 
On remplace ici la couche BiLSTM par une couche transformers.

 - 6 - [BERT - Fine-tuning](https://github.com/aminaghoul/NER-PyTorch/blob/master/6-Bert-fine-tuning-NER-CONLL.ipynb)
 
 Enfin, on implémente le modèle **BERT** décrit [ici](https://arxiv.org/abs/1810.04805) en utilisant [Hugging Face.](https://github.com/huggingface/transformers) 
 
 ## Résultats
 
 ## Références : 
 
 - [pytorch-sentiment-analysis ](https://github.com/bentrevett/pytorch-sentiment-analysis#tutorials) 
 
