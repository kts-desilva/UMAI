# About

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kts-desilva-prot-fs-streamlit-srcapp-eh8dfx.streamlit.app/)


This application provides an overview of the ovrain cancer proteomics data from CPTAC data portal. It is a dataset that provides protein expression profiles of ovarian tumor and control samples.

The app it is [deployed](https://kts-desilva-prot-fs-streamlit-srcapp-eh8dfx.streamlit.app/) in Streamlit.

The data were provided from the Clinical Proteomic Tumor Analysis Consortium (CPTAC) [source](https://cptac-data-portal.georgetown.edu/). 

You can check on the sidebar of the app:
- EDA (Exploratory Data Analysis)
- Feature Selection
- Model Prediction
- Model Evaluation

The predictions are made with classification performed to distinguish tumor vs control utlizing pre-trained machine learning models.

The data is available in raw format and pre-processed format as csv files inside the data directory.If you want to check the code, go through the notebook directory in this repository.

# Feature Selection

Feature selection based on recursive and iterative feature selection methods based on scikit-learn packages.
Machine learning algorithms used

- XGBoost
- Random Forest
- Support Vector Machine
- Stochastic Gradient Descent

# Model Definition

The structure of the training it is to wrap the process around a scikit-learn Pipeline. There were 4 possible combinations and 5 models, resulting in 20 trained models.

The combinations are regarding to perform Feature Creation and/or Target Transformations in the dataset.

Models:

- XGBoost
- Random Forest
- Support Vector Machine
- Stochastic Gradient Descent

Our main accuracy metric is RMSE. To enhance our model definition, we utilized Cross Validation and Random Search for hyperparameter tuning.
Further, we have considered using precision, recall and specificity metrics to access the quality of the developed methodlogy.

# Run the App

To run locally, clone the repository, go to the diretory and install the requirements.

```
pip install -r requirements.txt
```

Now, go to the src directory and run:

```
streamlit run app.py
```
