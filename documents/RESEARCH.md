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

| Model   | Link | Inputs | Outputs | Tokenizer | Checkpoints to Use | vocab_size | hidden_size | Image | num_hidden_layers | Can Run on Our EC2? |
| ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VisualBert | [Link](https://huggingface.co/docs/transformers/model_doc/visual_bert) | text embeddings, visual embeddings, visual token type ids, visual atten mask | TBD | BertTokenizer | ‘visualbert-vqa’ | 30522 | 768 | 512 visual_embedding_dim | 12 | TBD |
| CLIP (Doesn't do VQA out of box) | [Link](https://huggingface.co/docs/transformers/v4.21.3/en/model_doc/clip#usage) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| VILT | [Link](https://huggingface.co/docs/transformers/model_doc/vilt) | encodings from ViltProcessor | TBD | BertTokenizerFast | "dandelin/vilt-b32-finetuned-vqa" | 30522 | 768 | image_size 384 | 12 | TBD |
| LayoutMV2 | [Link](https://huggingface.co/docs/transformers/model_doc/layoutlmv2#transformers.LayoutLMv2ForQuestionAnswering) | ['input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'image'] encodings from LayoutLMv2Processor | TBD | LayoutLMv2TokenizerFast | LayoutLMv2ForQuestionAnswering | 30522 | 768 | image_size 128 | 12 | TBD |
| LayoutMV3 | [Link](https://huggingface.co/docs/transformers/model_doc/layoutlmv3) | input_ids, attention_mask, token_type_ids, bbox encodings from LayoutLMv3Processor  | TBD |  LayoutLMv3Tokenizer or  LayoutLMv3TokenizerFast | TFAutoModelForQuestionAnswering.from_pretrained("microsoft/layoutlmv3-base") | 50265 | 768 | image_size 128 | 12 | TBD |
| LXMERT | [Link](https://huggingface.co/docs/transformers/model_doc/lxmert) | input_ids, attention_mask, token_type_ids, bbox encodings from LayoutLMv3Processor  | TBD |  LayoutLMv3Tokenizer or  LayoutLMv3TokenizerFast | LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-base-uncased") but doc examples don't show image input only text, not sure if possible | 30522 | 768 | image_size 2048 | 12 | TBD |

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

### Things to Consider
1. Dataset doesn't split into train/test already so our testing dataset is going to be different from the original paper's but our metrics could still be used to compare if we randomly sample the same amount of data for testing as they did, will just need to caveat that it isn't orange to orange.
2. Some questions need annotations to reference and answer. We will need to include annotations straight from the get go. I think our first attempt can just be to feed the annotated image as the image input. Then we can try splitting original image and annotation inputs after we have something that kinda works and try to make it better.
3. Some data don't have questions like 1.png or 2.png. Looks like they are removed from the dataset in get_data_objects() so we good. We could maybe make our own question and answers if we really want to include them but I'm okay to not.
4. Okay so doing more research, looking at the data, and seeing the models available. I think a good way to take this project is part 1, to do the annotations straight onto the image and train the few transformers that we can and see results. Then for second part of project try to create our own processor/feature extractor that considers annotations maybe more separately from the image and more feature engineered to be useful to create these ['input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'image'] encodings that most of the models need and train on those and see. So the crux will be to understand how these processors work (ie: how tokenizers generate embeddings, how visual embeddings are created, how can we turn annotations into some embedding that we can combine with the other two to generate the input_ids, attention_masks, etc.)

