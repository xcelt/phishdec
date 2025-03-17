# Phishdec Project: Evaluating Phishing Email Classification Models

An artefact for the dissertation:

A Comparison between Traditional Naive Bayes and Ensemble Random Forest in Phishing Email Classification using Curated Datasets

By Xue Ling Teh

## Overview
### Datasets*
1. phishing
1. validate
1. enron 
1. ling
1. spamassassin
1. ceas_08
1. trec_05
1. trec_06
1. trec_07

*Oversized files have not been uploaded to GitHub remote due to the file size limit

#### Data preprocessing and cleaning
One of the following options:
* Drop empty rows with null, NA or NaN values

OR

* Impute/Fill NA or NaN values

### Model Types
1. Naive Bayes (NB): Multinomial
1. Random Forest (RF)

### Evaluation Metrics
* Accuracy
* Confusion Matrix
    * True Positive (TP)
    * True Negative (TN)
    * False Positive (FP)
    * False Negative (FN)    
* Classification Report: 
    * precision
    * recall
    * f1-score
    * support: total number of occurrences of a specific class in the dataset
* 5×2 Cross-Validation Paired T-test
    * t statistic
    * p-value

## Setup
### Prerequisites
* ~= Python 3.13.0

### Getting Started (Windows)
1. Create a virtual environment named venv: `python -m venv venv`
1. Install the required dependencies in `requirements.txt`:
    1. `python -m pip install -r requirements.txt`
1. Activate the virtual environment: `venv\Scripts\activate`
1. Run the program: `python main.py`

To zip up this project:
`git archive --format=zip --output ./artefact.zip HEAD`

## Notes
Filename convention: <i>title*_datasetnumber_randomstate.fileextension</i>

*Title with more than one word has extra underscores (_)

e.g. `nb_c_matrix_1_860.png` indicates a png image of a Naive Bayes Confusion Matrix of Dataset 1 with random state 860

Default random state is 42.

## References
Downey, A. (2022) _Think Bayes 2 — Think Bayes._ Available from: https://allendowney.github.io/ThinkBayes2/ [Accessed 19 October 2024].

Downey, A. (2024) _Think Python — Think Python._ Available from: https://allendowney.github.io/ThinkPython/ [Accessed 16 September 2024].

Chakraborty, S. (2023) _Phishing Email Detection._ DOI: https://doi.org/10.34740/kaggle/dsv/6090437.

Champa, A.I., Rabbi, M.F. & Zibran, M.F. (2024) Curated Datasets and Feature Analysis for Phishing Email Detection with Machine Learning. In: _2024 IEEE 3rd International Conference on Computing and Machine Intelligence (ICMI)._ April 2024 pp. 1–7. DOI: https://doi.org/10.1109/ICMI60790.2024.10585821.

Miltchev, R., Dimitar, R. & Evgeni, G. (2024) _Phishing validation emails dataset._ DOI: https://doi.org/10.5281/ZENODO.13474746.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M. & Duchesnay, É. (2011) Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research. 12(85): 2825–2830.

Toledo Jr, T. (2021) _Statistical Tests for Comparing Classification Algorithms._ 23 November 2021. Towards Data Science. Available from: https://towardsdatascience.com/statistical-tests-for-comparing-classification-algorithms-ac1804e79bb7/ [Accessed 12 March 2025]. 
