# George Washington University, Capstone - Fall 2022

![sample_diagram](https://github.com/alexiskaldany/CAP22FA/blob/main/example_data/0.png)

# Project Description

Improve on [research](https://arxiv.org/pdf/1603.07396.pdf)<sup>1</sup> done in multimodal task of Diagram Question Answering on the [AI2D Diagram Dataset](https://aws.amazon.com/marketplace/pp/prodview-ueiyrmcy4rzdm#usage)<sup>2</sup>.

## Table of Contents

1. [Team Members](#team_members)
2. [Folder Structure](#structure)
3. [Background and Related Works](#background)
4. [How to Run](#instructions)
5. [Architecture](#architecture)
6. [Results](#results)
7. [Presentation](#presentation)
8. [Paper](#paper)
9. [References](#references)
10. [Licensing](#license)

# <a name="team_members"></a>

## Team Members

* [Alexis Kaldany](https://github.com/alexiskaldany)
* [Joshua Ting](https://github.com/justjoshtings)

# <a name="structure"></a>

## Folder Structure

```
.
├── archive                 # Experimental scripts
├── assets                  # For storing other supporting assets like images, logos, gifs, etc.
├── documents               # Research documents
├── example_data            # Sample of dataset 
│ 
├── src                     # Main code directory
│   ├── models              # Directory for models
│   ├── tests               # Directory for doing R&D and code testings
│   ├── utils               # Directory for utility functions to support project
│ 
└── requirements.txt        # Python package requirements
```

# <a name="background"></a>

## Background and Related Works

### Original Paper

![sample_diagram](https://github.com/alexiskaldany/CAP22FA/blob/main/assets/background01.png)
Published on 24 Mar 2016, `A Diagram Is Worth A Dozen Images` set out to "study the problem of diagram interpretation and reasoning, the challenging task of identifying the structure of a diagram and the semantics of its constituents and their relationships" <sup>20</sup>


### Diagram Question Answering

Solving Diagram Question Answering (DQA) is a multimodal task that requires the model to understand the diagram and the question simultaneously. The general approach is:

  * Extract features from the diagram and question
  * Combine the features to generate a final representation
  * Use the final representation to predict the answer

DQA system can be seen as an algorithm that takes as input an image and a natural language question about the image and generates a natural language answer as the output.

A good DQA system must be capable of solving a broad spectrum of typical NLP and CV tasks, as well as reasoning about image content. It is clearly a multi-discipline AI research problem, involving CV, NLP and Knowledge Representation & Reasoning (KR). <sup>19</sup>

### What We Want to Try

![sample_diagram](https://github.com/alexiskaldany/CAP22FA/blob/main/assets/background04.png)

* We want to improve on the original paper by trying out more generalizable models and different approaches to the problem.

* We used a transformer-based model to extract and combine features from the diagram and question. The same model also predict the answer.



# <a name="instructions"></a>

## How to Run

This program is intended to be run on an EC2 configured with Ubuntu. The following instructions assume a fresh install.

### Requirements

* The dataset is contained on AWS S3. You may need AWS keys to download the dataset.
* Python 3.9 or higher.

### Setup

1. Clone the repo and enter into the directory

```bash
git clone https://github.com/alexiskaldany/CAP22FA.git
cd CAP22FA
```

2. Create the virtual environment and install the dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3. Run `prepare_and_download.py` to download the dataset and prepare the data for training. A `data` folder will be created inside `src` where all the data will be stored. This function also triggers the annotation scripts and takes care of all preprocessing.

```bash
python3 ./src/utils/prepare_and_download.py
```

#### Issues

There are small differences in the way paths work on different operating systems. Efforts have been taken to ameliorate this,

### Modeling

Execute model training for a specified model setup type:

```bash
cd src/models/
python3 model_training.py
```

or to execute as background program:

```bash
cd src/models/
nohup python3 model_training.py
ps ax | grep "model_training.py"
```

### Plot Results

Execute specified plot types for training and testing results:

```bash
cd src/models/
python3 plot_model_results.py
```

# <a name="architecture"></a>

## Architecture

### Environment Architecture

![sample_diagram](https://github.com/alexiskaldany/CAP22FA/blob/main/assets/env_architecture01.png)
![sample_diagram](https://github.com/alexiskaldany/CAP22FA/blob/main/assets/env_architecture02.png)

### Model Architecture

![sample_diagram](https://github.com/alexiskaldany/CAP22FA/blob/main/assets/model_selection.png)
![sample_diagram](https://github.com/alexiskaldany/CAP22FA/blob/main/assets/model_architecture01.png)

# <a name="results"></a>

## Results

![sample_diagram](https://github.com/alexiskaldany/CAP22FA/blob/main/assets/testing_results.png)

# <a name="presentation"></a>

## Presentation

[Final Presentation Slides](https://docs.google.com/presentation/d/1lfzdVxZWlUQ4vNnbCHOezjhFXg0yBI1lov0G_AoEURI/edit?usp=sharing)

# <a name="paper"></a>

## Paper

[Final Paper](https://docs.google.com/document/d/1F0sm1jjntVK7CECtQvxg2jkOu64l4hIXS2o6nf7ROYc/edit?usp=sharing)

# <a name="references"></a>

## References

1. [Github Repo](https://github.com/alexiskaldany/CAP22FA)
2. [A Diagram is Worth a Dozen Images](https://arxiv.org/pdf/1603.07396.pdf)

```
@article{Kembhavi2016ADI,
  title={A Diagram is Worth a Dozen Images},
  author={Aniruddha Kembhavi and Michael Salvato and Eric Kolve and Minjoon Seo and Hannaneh Hajishirzi and Ali Farhadi},
  journal={ArXiv},
  year={2016},
  volume={abs/1603.07396}
}
```

3. [AI2 Diagram Dataset (AI2D)](https://aws.amazon.com/marketplace/pp/prodview-ueiyrmcy4rzdm#usage)

```
AI2 Diagram Dataset (AI2D) was accessed on 9/5/2022 from https://registry.opendata.aws/allenai-diagrams.
```

4. Paper Code
5. [VISUALBERT: A SIMPLE AND PERFORMANT BASELINE FOR VISION AND LANGUAGE, Li et al., 2019](https://arxiv.org/pdf/1908.03557.pdf)
6. [ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision. Kim et al., 2021](https://arxiv.org/abs/2102.03334)
7. [LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. Huang et al., 2022](https://arxiv.org/abs/2204.08387)
8. [VisualBERT for Multiple Choice Hugging Face](https://huggingface.co/docs/transformers/v4.22.1/en/model_doc/visual_bert#transformers.VisualBertForMultipleChoice)
9. [VisualBERT for Question Answering Hugging Face](https://huggingface.co/docs/transformers/v4.22.1/en/model_doc/visual_bert#transformers.VisualBertForQuestionAnswering)
10. [VILT for Question Answering Hugging Face](https://huggingface.co/docs/transformers/model_doc/vilt#transformers.ViltForQuestionAnswering)
11. [LayoutMV3 Hugging Face](https://huggingface.co/docs/transformers/v4.22.1/en/model_doc/layoutlmv3)
12. [VisualBERT Demo](https://github.com/huggingface/transformers/blob/main/examples/research_projects/visual_bert/demo.ipynb)
13. [BERT Multiple Choice Sample](https://github.com/huggingface/transformers/blob/main/examples/pytorch/multiple-choice/run_swag.py)
14. [Fine Tuning on Multiple Choice Task](https://github.com/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb)
15. [Hugging Face](https://huggingface.co/models?pipeline_tag=visual-question-answering&sort=downloads)
16. [PyTorch AdamW Optimizer](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
17. [PyTorch Cross Entropy](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
18. Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dolla ́r, and C Lawrence Zitnick. Microsoft COCO captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015.
19. [Tryolabs](https://tryolabs.com/blog/2018/03/01/introduction-to-visual-question-answering)
20. [A Diagram is Worth a Dozen Images](https://arxiv.org/pdf/1603.07396.pdf)
# <a name="license"></a>

## Licensing

* MIT License
