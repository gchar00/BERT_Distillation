# Distillation

This repository is a fork of the Hugging Face's Distillation project(<a href="https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation">Original Project</a>).

Scripts used for evaluation forked as well from Hugging Face, you can find the code  <a href="https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification">here</a>!

# How to run

The following notebooks suggest a way to run this project on Kaggle and Google colab respectively:

<a href="https://www.kaggle.com/code/giorgosgotzias/distillation">Kaggle notebook</a>

<a href="https://colab.research.google.com/drive/1yPTiE7a65bqy6U56mDTv4oQ4i6LSE1fd?usp=sharing">Colab notebook</a>

# Changes

In distiller.py file we changed the loss used for a_mse flag, with a mse loss between attention layers of teacher and student (supposes the student has only 4 layers and teacher at least 12) for attention map transfer.

In lm_seqs_dataset.py we changed the lines 39 and 99 to explictitly define the type of arrays as object. That was necessary to avoid errors when you try to run training in google colab.

In scripts folder, we extended the original extract_distilbert.py to extract only four layers from BERT. We also implemented scripts to extract the first or last layers of Bert. Furthermore, you can find scripts to extract weights from DistilBERT.
