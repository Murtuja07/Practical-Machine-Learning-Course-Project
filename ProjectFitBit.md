---
title: "Practical Machine Learning-Project"
author: "Md Golam Murtuja"
date: "18 September 2018"
#output: html_document
output:
  html_document:
    keep_md: yes
---



#Overview
The goal of this assignment is to do a predictive analysis of a data set obtained from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They 
were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The data for this analysis was obtained from the following links:

The training data for this analysis is available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data is available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

We will perform some data clean up and build some prediction models using R's caret package to predict the "classe" variable on the text dataset of 20 test cases.

Data Loading and Cleanup


```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```


```r
library(ggplot2)
library(rattle)
```

```
## Rattle: A free graphical interface for data science with R.
## Version 5.2.0 Copyright (c) 2006-2018 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```


```r
library(rpart)

pml_train <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", na.strings = c("NA", "#DIV/0!", ""))
dim(pml_train)
```

```
## [1] 19622   160
```

```r
pml_test <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", na.strings = c("NA", "#DIV/0!", ""))
dim(pml_test)
```

```
## [1]  20 160
```

From the above, we can see that there are around 160 variables in the dataset and 19K observations. We also see that there are a lot of variables with NAs or blanks 
and so we can eliminate them as they will not help in the prediction modeling. We can also remove the first 7 cols which are basically names and timestamps which are 
not useful for modeling.


```r
library("caret")
# find variables with 90% NAs or blank values
badcols <- which(colSums(is.na(pml_train) | pml_train == "") > 0.9 * nrow(pml_train))
new_train <- pml_train[,-badcols]
# remove the first 7 columns
new_train <- new_train[,-c(1:7)]

## cleanup of test dataset

badcols2 <- which(colSums(is.na(pml_test) | pml_test == "") > 0.9 * nrow(pml_test))
new_test <- pml_test[,-badcols2]
# remove the first 7 columns
new_test <- new_test[,-c(1:7)]

# create a partition of the traning data set 
set.seed(9301)
ds1 <- createDataPartition(new_train$classe, p = 0.75, list = FALSE)
dsTrain <- new_train[ds1, ]
dsTest <- new_train[-ds1, ]

dim(dsTrain)
```

```
## [1] 14718    53
```

We have reduced the variables to just 53. We can use these 53 variables in our prediction modeling.

## Prediction Model Building
Since the prediction we want to do is that of classification of the type of exercise performed into A, B, C, D, E we will use the caret package's classification 
algorithms like Random Forests, classification tree and gradient boosting methods to build the models and compare the accuracy of the models in predicting the "classe" 
variable.

We will also use the caret packages k-fold cross validation algorithims to train the models.

# Model 1: Classification Tree

```r
# 5-fold cross validation
kcv <- trainControl(method="cv", number=5)
model1 <- train(classe ~ ., data = dsTrain, method = "rpart", trControl = kcv)

predict1 <- predict(model1, newdata = dsTest)

cm1 <- confusionMatrix(dsTest$classe, predict1)

cm1$overall["Accuracy"]
```

```
## Accuracy 
## 0.560155
```


```r
cm1$table
```

```
##           Reference
## Prediction   A   B   C   D   E
##          A 837 207 332  14   5
##          B 145 583 210  11   0
##          C  20  92 727  16   0
##          D  39 255 314 196   0
##          E   8 238 225  26 404
```


```r
fancyRpartPlot(model1$finalModel)
```

![](ProjectFitBit_files/figure-html/unnamed-chunk-7-1.png)<!-- -->


We can see that the accuracy with this model is around 56% and so not very accurate. We will explore other models.

# Model 2: Random Forests


```r
library("randomForest")
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:rattle':
## 
##     importance
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
model2 <- train(classe ~ ., data = dsTrain, method = "rf", trControl = kcv, verbose = FALSE)
predict2 <- predict(model2, newdata = dsTest)

cm2 <- confusionMatrix(dsTest$classe, predict2)

cm2$overall["Accuracy"]
```

```
##  Accuracy 
## 0.9924551
```


```r
cm2$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    0    0    0    0
##          B    4  943    2    0    0
##          C    0    8  846    1    0
##          D    0    0   18  785    1
##          E    0    0    0    3  898
```


```r
plot(model2$finalModel)
```

![](ProjectFitBit_files/figure-html/unnamed-chunk-10-1.png)<!-- -->
# Model 3: GBM - Gradient Boosting Method

```r
model3 <- train(classe ~ ., data = dsTrain, method = "gbm", trControl=kcv, verbose = FALSE)
predict3 <- predict(model3, newdata = dsTest)

cm3 <- confusionMatrix(dsTest$classe, predict3)

cm3$overall["Accuracy"]
```

```
##  Accuracy 
## 0.9596248
```


```r
cm3$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1377   13    3    1    1
##          B   34  883   31    1    0
##          C    0   27  812   14    2
##          D    2    3   26  769    4
##          E    1   11    3   21  865
```


```r
plot(model3)
```

![](ProjectFitBit_files/figure-html/unnamed-chunk-13-1.png)<!-- -->

# Conclusion
From the above three model accuracies, we can see that the Random Forests model predicts with the highest accuracy of 99.2%. The next best model is the GBM model with 
96%.

We can use this RF model(model2) to predict on the original Test dataset with 20 test cases.

```r
FinalPrediction <- predict(model2, new_test)

FinalPrediction
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
