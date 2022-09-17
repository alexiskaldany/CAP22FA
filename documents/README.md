# Research Dump

Lets use this for storing links and ideas for tackling the diagram set

* The most basic model would just be questions, answers, and images in the network, no annotations at all. Lets see if we can build that then explore adding in annotations. If we get that far, we can look into the graph neural network stuff

## Overview

So the data:

1. The images (.png)
2. The annotations (which include text, arrows, labels [TODO: more explanation of the annotations])
3. The questions (string)
4. The answers (4 strings)

### Input

For a full model, because the model needs to chose among the 4 answers, all 4 types of data would need to be placed into the input layer.

A nice explanation [here](https://huggingface.co/docs/transformers/main/en/tasks/multiple_choice) of how to create inputs when there are multiple choice answers / targets

### Output

Softmax will give relative weights among 4 choices.

## Multi-Channel Networks

### https://arxiv.org/pdf/1505.00468v6.pdf 

1. At 5.2 they describe how they integrate images and questions into a single network 

## Transformers

### Visual_Bert
link: https://huggingface.co/docs/transformers/model_doc/visual_bert

https://huggingface.co/docs/transformers/model_doc/visual_bert

### Examples
How to embed image: https://github.com/huggingface/transformers/blob/main/examples/research_projects/visual_bert/demo.ipynb

more indepth: https://github.com/huggingface/transformers/issues/13151

https://huggingface.co/docs/transformers/main_classes/data_collator

TODO: work through huggingface documentation

https://colab.research.google.com/drive/1bLGxKdldwqnMVA5x4neY7-l_8fKGWQYI?usp=sharing#scrollTo=5KPvzqT6mYJu generating visual embeddings

### CLIP
- Allows image and text
https://huggingface.co/docs/transformers/v4.21.3/en/model_doc/clip#usage

### VILT
ViLT incorporates text embeddings into a Vision Transformer (ViT), allowing it to have a minimal design for Vision-and-Language Pre-training (VLP).
https://huggingface.co/docs/transformers/model_doc/vilt

### LayoutMV2
LayoutLMv2 not only uses the existing masked visual-language modeling task but also the new text-image alignment and text-image matching tasks in the pre-training stage, where cross-modality interaction is better learned. 
https://huggingface.co/docs/transformers/model_doc/layoutlmv2#transformers.LayoutLMv2ForQuestionAnswering

### LayoutMV3
Experimental results show that LayoutLMv3 achieves state-of-the-art performance not only in text-centric tasks, including form understanding, receipt understanding, and document visual question answering, but also in image-centric tasks such as document image classification and document layout analysis.
https://huggingface.co/docs/transformers/model_doc/layoutlmv3

### LXMERT
It is a series of bidirectional transformer encoders (one for the vision modality, one for the language modality, and then one to fuse both modalities) pretrained using a combination of masked language modeling, visual-language text alignment, ROI-feature regression, masked visual-attribute modeling, masked visual-object modeling, and visual-question answering objectives. The pretraining consists of multiple multi-modal datasets: MSCOCO, Visual-Genome + Visual-Genome Question Answering, VQA 2.0, and GQA.
https://huggingface.co/docs/transformers/model_doc/lxmert


### Sept 11 Findings

- After converting image to tensor, need to get a flat vector of equal length to tokenized question + possible answers

Possible plan:
1. 