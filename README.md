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
│   ├── utils               # Directory for utility functions
│   └── ...                 # Lorem ipsum
└── ...
```

# <a name="background"></a>
## Background and Related Works
lorem ipsum - once we have more of a concrete plan we can start populating this that describes what's been done, why we're doing this, what type of things ppl tried etc.

# <a name="instructions"></a>
## How to Run

### Setup

# <a name="architecture"></a>
## Architecture
lorem ipsum - for descriptions of both our cloud/software architecture and model architecture. will fill towards the end of project and can add some visual diagrams to help

# <a name="presentation"></a>
## Presentation

# <a name="paper"></a>
## Paper

# <a name="references"></a>
## References
1. [A Diagram is Worth a Dozen Images](https://arxiv.org/pdf/1603.07396.pdf)
```
@article{Kembhavi2016ADI,
  title={A Diagram is Worth a Dozen Images},
  author={Aniruddha Kembhavi and Michael Salvato and Eric Kolve and Minjoon Seo and Hannaneh Hajishirzi and Ali Farhadi},
  journal={ArXiv},
  year={2016},
  volume={abs/1603.07396}
}
```
2. [AI2 Diagram Dataset (AI2D)](https://aws.amazon.com/marketplace/pp/prodview-ueiyrmcy4rzdm#usage)
```
AI2 Diagram Dataset (AI2D) was accessed on 9/5/2022 from https://registry.opendata.aws/allenai-diagrams.
```
3. 

# <a name="license"></a>
## Licensing
* MIT License

## MISC to put somewhere later

### Data Standard Structure

- `get_data_objects` in `src.utils.prepare_and_download` creates a list where each element is a list of three dicts:
    1. The first dict is simply`{"image_path": "{DATA_DIRECTORY}/src/data/ai2d/images/0.png"}`. Use this to get the path to the image
    2. The second object is the entire annotations json for that image.
    >TODO: create functions that break down annotations into more useful elements
    3. The third object is the entire questions json for that image.
    >TODO: There can be multiple questions per images, so we'll need a function that breaks down this dictionary into a list with each element being a question 

### Models

- We should create a standard

### Work done so far

- Finished the annotation functions
  1. Could use some touchups for the arrowheads
- Close to finishing the functions which build the unified csv containing all the data the models need for training
- Learned how to extract visual embeddings and how to work more closely with pre-trained models.
