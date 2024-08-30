# classification-suite

This repository contains the software required to reproduce the data in [Data Efficiency of Classification Strategies for Chemical and Materials Design](https://doi.org/10.26434/chemrxiv-2024-1sspf).

The `classification-suite` repository contains the `ClassificationSuite` Python package which makes it easy to run active learning and space-filling classification algorithms with a wide array of samplers and models. The package also contains 31 classification tasks from the chemical and materials science literature. To implement your own sampler, just add a method to ``sample.py''. To implement your own model, add a class in the `Models/` directory that inherits from the `AbstractModel` API in `model.py`. Tasks can also be added by including your task as a numpy binary file in `Tasks/Datasets/` where labels are 1 or -1 and stored in the last column. Installation and usage instructions are included below. 

### Installation

### Usage
