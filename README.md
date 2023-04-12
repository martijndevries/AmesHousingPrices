# Project 2 - Ames Housing Data and Kaggle Challenge

by Martijn de Vries </br>
martijndevries91@gmail.com

## Problem Statement

A real estate company in Ames, Iowa is looking for a new and improved way to evaluate the market value of a house. Using the Ames data set as training data, we will build a predictive linear regression model to predict the sale price of a house as well as possible.

To gauge the model performance, I will compare my results against a 'benchmark model', which is a simple OLS regression of total living area versus sale price. How much can a more complex model improve over this simple basic model? I will try out different models with different numbers of features using different linear regression techniques, and ultimately identify which model does the best job at predicting the market value of the house.

 
## Repository Overview
    
This repository consists of the following:

<ol>
   <li> The directory <code>./code</code> contains the four notebooks notebook used for this analysis. In data_cleaning.ipynb, I do basic EDA and data cleaning of the data, checking for missing values and doing imputations. In feature_engineering.ipynb, I take a closer look at the data, try to figure out which features could best be used to model the sale price, and build 'model input' files for increasingly complex models. The model input files can be used for modeling with minimal processing. In modeling.ipynb, I load in these model input files and use them to build a variety of models. I use the r2 score, the cross-validation score, and the RMSE to evaluate which model does the best job at predicting sale prices. In modeling_insights.ipynb, I hone in on the best model(s), evaluate them in more detail, and compare them against the benchmark model. </li>
   <li> The directory <code>./data</code> contains the input data csv files, train.csv and test.csv. It also includes the cleaned train and test data, as well as the Kaggle contest submission files, which consist of a set of housing IDs and predicted sale prices for those houses, using various models from the modeling notebook.
    <li> The directory <code>./model_inputs</code> contains the model input files constructed in feature_engineering.ipynb. These files can be loaded in and used to fit the model with minimal processing.
   <li> The directory <code>./figures</code> contains all the figures that are made during the analysis in the notebooks, in .png formates </li>
    <li> The slides for the project presentation are in the file <code>project2_martijn_slides.pdf</code> </li>
</ol>


## Data Overview

For this project, I used the Ames housing data set, which was pre-split into a train and test data set. The training set consists of 2051 houses and 81 columns, including the sale price. The test set consists of 878 houses, and has the exact same columns exluding the sale price column. The data consists of both numerical columns (eg. the square footage of the land, the number of bathrooms, or the number of fireplaces), and categorical columns (eg. is there a pool, what kind of garage is there, does the property have a fence). The figure below shows a histogram of the target variable, the sale price



## Data Cleaning

After general inspection of the data, we looked at missing values. The image below shows all the columns in which missing values were detected.

<img src="./figures/missingvals.png" style="float: left; margin: 20px; height: 400px">

I dealt with the missing values in a few different ways:

<ol>
    <li> A few of the columns have 'NA' (Not applicable) as a category. Eg. for the 'Alley' column, some houses do not have an Alley and thus have 'NA' filled in. Pandas erroneously interprets this as a NaN when loading in the data. In these cases, I simply imputed the value 'NP' (Not present) </li>
    <li> 
</ol>