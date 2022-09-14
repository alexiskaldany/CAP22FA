# CAP22FA
Fall 22` Capstone Project, by Josh Ting and Alexis Kaldany


## Setup

This project is set up using pyenv and poetry. These allow you to easily customize which version of python to use and to manage dependencies respectively.

If you don't already have both in place:

- Follow the instructions for pyenv at https://github.com/pyenv/pyenv#installation. Please do not skip the
  pre-requisites step. There is also more info at https://github.com/pyenv/pyenv/wiki#suggested-build-environment to
  ensure you get this right. There is a dated but nice tutorial at https://realpython.com/intro-to-pyenv if you
  want to learn
  more about pyenv.
- Follow the instructions to install poetry at https://python-poetry.org/docs/#installation
  - Now configure poetry so that the virtualenv will be created within the repository when the project is installed:
    `poetry config virtualenvs.in-project true`

- Then build the virtual environment with:

```bash
   poetry install
```
