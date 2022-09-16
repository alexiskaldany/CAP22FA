# CAP22FA
Fall 22` Capstone Project, by Josh Ting and Alexis Kaldany


## Setup

This project is set up using pyenv and poetry.

- Follow the instructions for pyenv at https://github.com/pyenv/pyenv#installation. Please do not skip the
  pre-requisites step. There is also more info at https://github.com/pyenv/pyenv/wiki#suggested-build-environment 
- Follow the instructions to install poetry at https://python-poetry.org/docs/#installation
  - Now configure poetry so that the virtualenv will be created within the repository when the project is installed:
    `poetry config virtualenvs.in-project true`

- Then build the virtual environment with:

```bash
   poetry install
```

## Data Standard Structure

- `get_data_objects` in `src.utils.prepare_and_download` creates a list where each element is a list of three dicts:
    1. The first dict is simply`{"image_path": "{DATA_DIRECTORY}/src/data/ai2d/images/0.png"}`. Use this to get the path to the image
    2. The second object is the entire annotations json for that image.
    >TODO: create functions that break down annotations into more useful elements
    3. The third object is the entire questions json for that image.
    >TODO: There can be multiple questions per images, so we'll need a function that breaks down this dictionary into a list with each element being a question 

## Models

- We should create a standard


## Work done so far

- Finished the annotation functions
  1. Could use some touchups for the arrowheads
- Close to finishing the functions which build the unified csv containing all the data the models need for training
- Learned how to extract visual embeddings and how to work more closely with pre-trained models.
