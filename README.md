This repository contains the code produce for the "Estimating Marketing Uplifts as Heterogeneous Treatment Effects with Meta-learners" team of the Study Week on Causal Inference for Industry held in the Pascale institute in July 2023. 

Our work was based on the meta-learners described in the following article:

    "Comparison of meta-learners for estimating multi-valued treatment heterogeneous effects"
    By Naoufal Acharki, Ramiro Lugo, Antoine Bertoncello, Josselin Garnier
    Archive preprint: https://arxiv.org/abs/2205.14714

The core contribution of possible general interest are the implementations of multiple metalearners from the litterature. More precisely:
- S learner
- T learner
- M learner
- X learner
- R learner
- DA learner
- DR learner

No clean command line interface to run these implementations was developped in the limited time at our disposal. Using our implementations for specific other purposes will likely require to understand our code to some extent.
To test eveyrthing and as an entry point, run "runAllLearners.py" in python3. This will run all learners on the (synthetic) dataset put in data.csv and put the result in the "output" directory. 

Additional information on the project and context during which this code was produced can be found in the "ProjectGoal.pdf" file.

# Requirements
- Install causalml with pytorch by following the following instructions https://causalml.readthedocs.io/en/latest/installation.html
- Install sklearn through pip: 
> pip install scikit-learnpip 
- Install econml through pip:
> pip install econml
- Install numpy through pip
> pip install numpy
- Install pandas through pip
> pip install pandas

# Code explanation
Various metalearners are defined in the files contained in the "MetaLearners" directory. 
Each are called with the "trainAndPredict" function, which always has the same signature (see comments in code).
The files "data_slicer.py" and "Viz.py" were used to produce visualisations of results in multiple ways.

# Acknowledgements
Many meta learner implementations rely directly on their implementation in the econml library.
We are grateful for the all the work by third parties that we relied upon.
We are especially grateful to the company Ekimetrics, which provided the original dataset (not public) and proposed this topic.