
# A new paradigm in lignocellulolytic enzyme cocktail optimization: Free from expert-level prior knowledge and experimental datasets

Our methodology draws inspiration from the modeling techniques of protein sequences and compound 1D representations, using intricate neural architectures. a, We introduce the EA-net structure. a-1, The model receives 9 distinct features as input. a-2, Central to EA-net are attention and embedding blocks, augmented with a residual connection block that enhances the precision by approximately 2%. a-3, The output is adaptable, catering to both supervised and unsupervised objectives. b, We detail the contrastive learning technique we employed. c, Our multi-clustering approach is also depicted. Specific details refer to [paper](https://www.sciencedirect.com/science/article/abs/pii/S0960852423011860)
![image](figures/model.png)

# Installation
## DATA

To quickly get started and understand the functionality of the code, we provide a dataset example [data](data). It includes a portion of the original data, allowing you to easily run and test the code, and more data can be referenced in the paper.

## Requirements
You'll need to install following in order to run the codes. Refer to [environment.yaml](environment.yaml) for a conda environment tested in Linux.
tensorflow == 1.14
numpy == 1.19.2
matplotlib == 3.3.4
pandas == 1.1.5
scikit-learn == 0.24.2 

# Usage

## Data generation
Running the following code can process the raw data and convert it into a form that the code can run, which will be displayed in the middle_ Generate three files under the result file: me2id.txt, labels2id2cnt.txt, and train.txt
```python
python make_data.py
 ```
## Train the model
Training command:
```python
 python main.py train
```
## Model testing
Running the following command will generate the intermediate layer for the last round of model training in the form of an encode.txt file, in preparation for the next clustering:
```python
python main.py test
```
# clustering
In the [clustering](clustering) file, there is the multi method voting clustering code. Running the following command in a Linux window can obtain the results in the [acc](clustering/acc) file:
```python
sh run.sh
```

**For citation:**

@article{GAO2023129758,
title = {A new paradigm in lignocellulolytic enzyme cocktail optimization: Free from expert-level prior knowledge and experimental datasets},
journal = {Bioresource Technology},
volume = {388},
pages = {129758},
year = {2023},
}