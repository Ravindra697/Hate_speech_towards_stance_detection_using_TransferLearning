# Hate_speech_towards_stance_detection_using_TransferLearning

## Hate speech detection
Hate speech detection is the process of identifying and flagging language that is considered offensive or harmful towards individuals or groups based on their 
characteristics such as race, gender, sexual orientation, religion, or ethnicity. This type of language can take many forms, including but not limited to, derogatory
language, slurs, insults, threats, or calls for violence.

## Transfer Learning
Transfer learning is a machine learning technique where a pre-trained model is used as a starting point for a new task. In transfer learning, the knowledge gained 
from learning one task is transferred to another related task, resulting in faster and more accurate learning.

The pre-trained model has already learned from a large dataset, often involving a general task like image classification, speech recognition, or natural 
language processing. This pre-training makes the model familiar with the features and patterns of the data, which can be useful for learning new tasks 
with smaller datasets. The pre-trained model can be used as a feature extractor to generate a new set of features that are specific to the new task.

During the fine-tuning process, the pre-trained model is further trained on a new dataset of labeled examples to adapt it to the specific task. 
Fine-tuning adjusts

## BERT (Bidirectional Encoder Representations from Transformers)
BERT is a pre-trained language model developed by Google, which has gained significant attention for its 
exceptional performance on various natural language processing (NLP) tasks. BERT is a transformer-based model that has been pre-trained on a large corpus of text 
data using an unsupervised learning technique called masked language modeling. This pre-training enables BERT to understand the context of words in a sentence and 
capture complex relationships between them.

## Work flow
### Required modules
  * import pandas as pd
  * import numpy as np
  * from tqdm.auto import tqdm
  * import tensorflow as tf
  * from transformers import BertTokenizer
  * import matplotlib.pyplot as plt
  * from transformers import TFBertModel
