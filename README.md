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
Original Paper:
![sample_diagram](https://github.com/alexiskaldany/CAP22FA/blob/main/assets/background01.png)

What We Want to Try:
![sample_diagram](https://github.com/alexiskaldany/CAP22FA/blob/main/assets/background04.png)


# <a name="instructions"></a>
## How to Run

### Setup

# <a name="architecture"></a>
## Architecture

Environment Architecture
![sample_diagram](https://github.com/alexiskaldany/CAP22FA/blob/main/assets/env_architecture01.png)
![sample_diagram](https://github.com/alexiskaldany/CAP22FA/blob/main/assets/env_architecture02.png)

Model Architecture
![sample_diagram](https://github.com/alexiskaldany/CAP22FA/blob/main/assets/model_selection.png)
![sample_diagram](https://github.com/alexiskaldany/CAP22FA/blob/main/assets/model_architecture01.png)

# <a name="presentation"></a>
## Presentation
[Final Presentation Slides](https://docs.google.com/presentation/d/1lfzdVxZWlUQ4vNnbCHOezjhFXg0yBI1lov0G_AoEURI/edit?usp=sharing)

# <a name="paper"></a>
## Paper
[Final Paper]()

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

# <a name="license"></a>
## Licensing
* MIT License