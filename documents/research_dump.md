# Research

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