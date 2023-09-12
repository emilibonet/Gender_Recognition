# Age and Gender Recognition In The Wild

_Author_: Emili Bonet i Cervera

_Description of the project_:

The estimation of demographics from image data, and in particular of gender and age, is a subject with an extensive amount of applications. However, current state-of-the-art is not entirely focused on realistic and unconstrained scenarios, which makes those approaches unusable for certain real-life settings. This thesis analyzes the issue of robust age and gender prediction, and proposes a new paradigm to build upon an alternative framework from which methods that are more capable in realistic situations can be developed. Namely, we present a method based on Deep Neural Networks (DNNs) that acts as an ensemble model, including predictions from both corporal and facial features. Thus, our model can act both when faces are not very visible or are occluded, and can take advantage of the extra information when they are visible. The system presented combines multiple off-the-shelf models such as RetinaFace and ShuffleNet for facial tasks, and Faster R-CNN with ResNet backbone pre-trained on COCO for human detection. From my side, a module was trained to predict gender and age based on body detections, where EfficientNet is used as backbone. Consequently, it was demonstrated that body-based models have the capacity to be more resilient.


## Setting up the Environment

Conda is used to set up the environment. In the main directory of this project, there is a YAML file named `conda_environment.yml` that contains all the dependencies that need to be installed. This environment is created by running the following command:

```
conda env create -f conda_environment.yml
```

After this installation, activate the `gerec` environment with the command

```
conda activate gerec
```

and finally test that it works by running the example code in the next section.

## Running the example code

In the `codes` directory there is a notebook file named `example.ipynb` that provides an example of how to use the main class of this project, the `EnsembleModel`.

## Data management

The data is allocated in the `data` directory. This directory, in turn, ought to contain two more folders: `body`, for the datasets that deal with body detections, and `face`, for the datasets that deal with facial detections.

There are three main pipelines used to manage the data.

1. __Pathset creation__

By running the file `pathset_creation.py`, you'll be able to automatically create a pathset in the data directory. A pathset is nothing more than a standarized directory for the data which will help manage it for the later pipelines. Each pathset contains a `imgpaths.txt` file with absolute paths to all the images, and an `annotations` directory with the annotation files for each dataset.

2. __Synthetic annotations__

Once the environment is ready, you'll dispose of different off-the-shelf models that we use to generate the synthetic data. To generate these data, run the `pseudo_labelling.py` file.

3. __Curated data reformatting__

After obtaining the curated data, it is necessary to transform it back to our format. To do so, the file curated_pipeline.py is used. This file looks for the directory with the curated data at `{root}/data/body/curated`. Change this string if the directory is allocated somewhere else.
