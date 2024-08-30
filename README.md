# classification-suite

This repository contains the software required to reproduce the data in [Data Efficiency of Classification Strategies for Chemical and Materials Design](https://doi.org/10.26434/chemrxiv-2024-1sspf).

![ClassificationSuite Summary](methods.png)

The `classification-suite` repository contains the `ClassificationSuite` Python package which makes it easy to run active learning and space-filling classification algorithms with a wide array of samplers and models. The package also contains 31 classification tasks from the chemical and materials science literature. To implement your own sampler, just add a method to `sample.py`. To implement your own model, add a class in the `Models/` directory that inherits from the `AbstractModel` API in `model.py`. Tasks can also be added by including your task as a numpy binary file in `Tasks/Datasets/` where labels of 1 or -1 and stored in the last column. Installation and usage instructions are included below. 

### Installation

You can install the `ClassificationSuite` Python package in a new conda environment using the following commands:
``` 
conda deactivate
conda env create -n class-env
conda install python=3.11
git clone https://github.com/webbtheosim/classification-suite.git
cd classification-suite
pip install -r requirements.txt
pip install -e .
```

### Usage

To apply a classification algorithm to a given task, you can use `ClassificationSuite/src/run.py`. An example is included below. 
```
cd ClassificationSuite/src/
python run.py --scheme al --task princeton --sampler random --model gpc_ard --seed 42 --results_dir PATH_TO_RESULTS --visualize
```
