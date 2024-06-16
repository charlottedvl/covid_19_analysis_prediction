# Analysis on a Covid-19 dataset
## Table of content 

- [Context](#context)
- [Formatting](#formatting)
- [Bayesian Network](#bayesian-network-)
- [Machine Learning models](#machine-learning-models)

## Context

This repository tries to analyze a dataset concerning the COVID-19 quick spread. 
The data used can be found on this repository: https://github.com/beoutbreakprepared/nCoV2019/tree/master

## Formatting

The dataset contains much information and is quite complete. 
Unfortunately, all columns are not usable in the actual state, as most of them are string and many rows have null value. 
The objective is to format the data so that performing the analysis becomes simpler. We then work with the [cleaned dataset](data/cleaned_dataset.csv).
The final cleaned dataset has 307382 rows and the following features:

<p align="center">
  <img height="200" src="data/analysis/empty_rows_cleaned_dataset.png" alt="Cleaned dataset analysis" />
</p>

More information about how the dataset have been formatting can be found here: [the format data markdown](format_data/README_formatting.md).

## Analysis 

Now that our dataset is ready to be used, we can perform multiple analysis on it. 
We choose to compute a correlation matrix, focusing on the target feature: the outcome of the patient.  

<p align="center">
  <img height="200" src="data/analysis/correlation_matrix.png" alt="Correlation matrix" />
</p>

We also create scatter plots concerning the features 'outcome' and 'age' and can be found here: [scatter plots](data/scatter_plots).

Finally, we perform a PCA restraining to the features most correlated to the target value: the outcome. 

<p align="center">
  <img height="220" src="data/analysis/pca_plot.png" alt="PCA plot" />
</p>

## Machine Learning models

The next step is to use this dataset to try to use this dataset to make some prediction. 
We are going to use 1 graphical model - the Bayesian Network - 1 linear regression model and 2 machine learning models to do so: K-Nearest-Neighbors and K-Means. 

More information about each model in the [model README](models/README_models.md).

### Bayesian Network

First, we are going to compute some probabilities thanks to the following Bayesian Network. 

<p align="center">
  <img height="250" src="data/bayesian_network/bayesian_network.png" alt="Bayesian Network used" />
</p>

We compute the probability for someone to have symptoms if the person visited Wuhan:  

<p align="center">
  <img height="150" src="data/bayesian_network/have_symptoms_visited_wuhan.png" alt="P(having_symptoms | visited_Wuhan = yes)" />
</p>
![P(having_symptoms | visited_Wuhan = yes)](data/bayesian_network/have_symptoms_visited_wuhan.png)

The probability for someone to have symptoms if the person visited Wuhan: 

<p align="center">
  <img height="200" src="data/bayesian_network/true_patient_visited_wuhan_have_symptoms.png" alt="P(true_patient | visited_Wuhan = yes, have_symptoms = yes)" />
</p>

And the probability for a person to be a true patient if this person has symptoms of
COVID-19 and this person visited Wuhan:  

<p align="center">
  <img height="200" src="data/bayesian_network/outcome_visited_wuhan.png" alt="P(die | visited_Wuhan = yes)" />
</p>

We can also predict the average recovery interval for a patient if this person visited Wuhan:

<p align="center">
  <img width="500" src="data/bayesian_network/average_recovery_time.png" alt="Average recovery time if visited Wuhan" />
</p>

### K-Nearest-Neighbors (KNN)

We use the KNN model to predict patients' outcome. A lot of them are still hospitalized, and it could be interesting 
to predict their outcome based on the dataset other features. We divide the dataset in two different: 
one for patients whose outcome is known, one for those whose outcome isn't known. 

We train our model with the first dataset and test the accuracy and the confusion matrix on the results obtained with this dataset. 
We obtain: 

<p align="center">
  <img height="100" src="data/knn/accuracy_confusion_matrix.png" alt="Average recovery time if visited Wuhan" />
</p>

Then, we use our model to predict the outcomes of the dataset composed of hospitalized patients. 
We can compare the repartition of death and recovery predicted by the model with the actual proportion of the first dataset. 
As a reminder, 0 is death and 2 recovery. We obtain:

<p align="center">
  <img height="100" src="data/knn/outcomes.png" alt="Actual outcomes" style="margin-right: 10px;" />
  <img height="100" src="data/knn/predicted_outcomes.png" alt="Outcomes predicted by the model" />
</p>

We can notice that the repartition is quite similar. 

### Linear regression model

There are 272947 patients who don't have an age, while it is important information for a person. 
We use a linear regression model to predict the age of the patients.

We perform a first linear regression using only the two features that are most correlated with age 
and compute the mean square error:

<p align="center">
  <img width="200" src="data/linear_regression/mean_square_error.png" alt="Mean square error" />
</p>

As the score is not that good, we try to improve it. To do so, we suppress columns that have more
than 90% of missing values and choose the 3 most correlated features. 
This allows us to improve the mean square error score that we have.

<p align="center">
  <img width="200" src="data/linear_regression/mean_square_error_improved.png" alt="Mean square error improved" />
</p>

This helps us to get better results but if we look closely, we could see that the variance of the
predicted age is very low and most predicted age are around 33 years old.
Some improvements can still be made to get better results. 

<p align="center">
  <img width="700" src="data/linear_regression/analysis_regression.png" alt="Analysis on the predicted age" />
</p>

### K-Means

Now, we want to separate the data in clusters. To do so, we use the K-Means model. 
We analyze our results with the silhouette score of the clustering. We obtain 0.99 which is pretty good.

<p align="center">
  <img height="220" src="data/k_means/k_means_plot.png" alt="Clusters" />
</p>
