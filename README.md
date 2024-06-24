# Language Detection with Naive-Bayes & Support Vector Algorithm
## Overview
In this short repository, I implemented a Language Multiclassification and detection algorithm uing Multinomial Naive Bayes and OvO & OvR extended classification Support Vector Machines. This model is trained on 2 different datasets which encompasses 30 distinct popular languages worldwide.
## Dataset
```dataset.csv``` and ```Language Detection.csv``` are two different datasets which both consist of a Text and Language column.
## Results
1. Multinomial NB Model Score: ```94.57 %```
2. OvO (One vs. One) & OvR (One vs. Rest) SVM Model Score: ```84.48 %``` 
3. Example Run:
```
Enter text to detect:  Apa namamu?
Text is Indonesian
```
## Conclusion
Given the high accuracy when implementing both Machine Learning models, this model does a good job at classifying languages, but only those (30 languages) that are available at the two datasets used. Further improvements to this project could include a more diverse and larger dataset which could encompass more languages worldwide.

