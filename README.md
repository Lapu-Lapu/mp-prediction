Evaluating Perceptual Predictions based on Movement Primitive Models in VR- and Online-Experiments
==============================

Dataset and analysis software associated with the following publication:

[1] Benjamin Knopp, Dmytro Velychko, Johannes Dreibrodt, Alexander C.  Schütz, and Dominik Endres. 2020. Evaluating Perceptual Predictions based on Movement Primitive Models in VR- and Online-Experiments .  In ACM Symposium on Applied Perception 2020 (SAP ’20), September 12– 13, 2020, Virtual Event, USA. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3385955.3407940


## Requirements:
- Python distribution [`Anaconda or Miniconda`](https://docs.conda.io/en/latest/miniconda.html)
- `Unix-like OS` (tested on Ubuntu 16.04, Arch Linux, OpenSUSE)
- (optional) [`git lfs`](https://github.com/git-lfs/git-lfs/wiki/Installation) if you want to clone this repo from github

## Instructions

The raw data is stored in directory `data/raw`. Please replicate the computing environmen
using conda in an Unix-like operating system. Use the Makefile to process the raw data
through intermediate results into plots shown in the publication. The preprocessing is
slightly different, therefore the results will differ in an insignificantly.

1. Make sure to replicate the python environment using conda:

    - `conda env create -f environment.lock.yaml --force`
    - `conda activate mp_prediction`
    - `pip install -e .`

2. Run `make all`. This will take a few minutes to save all figures
   presented in [1] in folder `reports/figures`. You can also run processing
   scripts separately. Please refer to the `Makefile` to check dependencies of
   the processing, training, and visualization scripts.

If you have trouble executing these steps, please contact me: benjamin.knopp@uni-marburg.de.


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
