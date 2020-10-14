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

 - 1 - [BiLSTM-NER](https://github.com/aminaghoul/NER-PyTorch/blob/master/1-BiLSTM-NER.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb)
 
 Dans ce premier notebook, le modèle utilisé est un LSTM bidirectionnel classique pour faire de la reconnaissance d'entités nommées (NER).

 - 2 - [BiLSTM+Embedding de caractère-NER](https://github.com/aminaghoul/sentiment-analysis/blob/master/0-MachineLearning.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb)
 
Ici, on ajoute au modèle précédent un embedding de caractères en utilisant des convolutions.

 - 3 - [BiLSTM+CRF](https://github.com/aminaghoul/sentiment-analysis/blob/master/0-MachineLearning.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb)
 
 On ajoute au modèle précédent une couche CRF.
 
 - 4 - [FastText](https://github.com/aminaghoul/sentiment-analysis/blob/master/0-MachineLearning.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb)

Le modèle **FastText** issu de l'article [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759) est implémenté ici.

 - 5 - [CNN](https://github.com/aminaghoul/sentiment-analysis/blob/master/0-MachineLearning.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb)
 
On utilise le modèle **CNN** pour l'analyse de sentiments.
 
 - 6 - [AttentionLSTM](https://github.com/aminaghoul/sentiment-analysis/blob/master/0-MachineLearning.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb)
 
 On utilise un modèle **LSTM** combiné avec de l'**attention**. Ce modèle est décrit dans l'article [Text Classification Research with Attention-based Recurrent Neural Networks.](https://www.researchgate.net/publication/323130660_Text_Classification_Research_with_Attention-based_Recurrent_Neural_Networks)
 
 - 7 - [TransformersLSTM](https://github.com/aminaghoul/sentiment-analysis/blob/master/0-MachineLearning.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb)
 
On utilise la libraire **transformers** pour importer un modèle transformer pré-entraîné pour obtenir les embedding du texte, et les utiliser dans un modèle **LSTM** pour prédire le sentiment.
 
 - 8 - [Transformers](https://github.com/aminaghoul/sentiment-analysis/blob/master/0-MachineLearning.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb)

Dans ce notebook est implémenté le modèle **transformer** en PyTorch décrit dans l'article [Attention Is All You Need](https://arxiv.org/pdf/1706.03762v5.pdf)
 
  - 9 - [BERT](https://github.com/aminaghoul/sentiment-analysis/blob/master/0-MachineLearning.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb)

 Enfin, on implémente le modèle **BERT** décrit [ici](https://arxiv.org/abs/1810.04805) en utilisant [Hugging Face.](https://github.com/huggingface/transformers) 
 
 ## Résultats
 
 ## Références : 
 
 - [pytorch-sentiment-analysis ](https://github.com/bentrevett/pytorch-sentiment-analysis#tutorials) 
 
