# Toxic_Commnets_Classification

***DISCLAIMER AND WARNING: Because of the nature of this project, this repository contains words that are very offensive. Needless to say, it is not the intention of the author of this report, and this project, to offend anybody.***

Social media has been growing fast and profound. The percentage of US adults who use social media increased from 5% in 2005 to 79% in 2019.  The rapid growth of social media has raised serious concerns about the threat of abuse and harassment on these platforms. Social media platforms struggle to facilitate conversations effectively and may lead to limiting or completely blocking user comments. 
As mentioned in the Kaggle competition from where the dataset used in the project was obtained: “The conversation AI team, a research initiative founded by Jigsaw and Google (both a part of Alphabet), is working on tools to help improve the online conversation.” One area of focus for this team is the study of negative online behaviors, like toxic comments (i.e., comments that are rude, disrespectful, or otherwise likely to make someone leave a discussion).
This capstone project aims to classify comments into non-toxic and toxic groups. In this regard, the main goals of this project are:

- To create models to classify comments into non-toxic and toxic.
- To perform an interpretability analysis and specify features impacting each class of non-toxic and toxic.

The dataset used in this project is obtained from the Kaggle website and includes labeled Wikipedia comments. We trained 20 models to classify the comments as non-toxic and toxic. Moreover, we used BayesianSearchCV to perform hyperparameter tuning of the trained models. In terms of toxic class, our best model for f1-score was the LGBM classifier with an f1-score of 0.68. Also, our best model for recall parameter was Logistic Regression in conjunction with Random Undersampling with a score of 0.82. 
