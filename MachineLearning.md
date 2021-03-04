# Practical Machine Learning Coursera Peer Assessment  

## Summary

This report uses machine learning algorithms to predict the manner in which users of exercise devices exercise. 


### Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website [here:](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset). 

### Data 

The training data for this project are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


### Set the work environment and knitr options


```r
rm(list=ls(all=TRUE)) #start with empty workspace
startTime <- Sys.time()

library(knitr)
opts_chunk$set(echo = TRUE, cache= TRUE, results = 'hold')
```

### Load libraries and Set Seed

Load all libraries used, and setting seed for reproducibility. *Results Hidden, Warnings FALSE and Messages FALSE*


```r
library(ElemStatLearn)
library(caret)
library(rpart)
library(randomForest)
library(RCurl)
set.seed(2014)
```

### Load and prepare the data and clean up the data




Load and prepare the data


```r
trainingLink <- getURL("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
pml_CSV  <- read.csv(text = trainingLink, header=TRUE, sep=",", na.strings=c("NA",""))

pml_CSV <- pml_CSV[,-1] # Remove the first column that represents a ID Row
```

### Data Sets Partitions Definitions

Create data partitions of training and validating data sets.


```r
inTrain = createDataPartition(pml_CSV$classe, p=0.60, list=FALSE)
training = pml_CSV[inTrain,]
validating = pml_CSV[-inTrain,]

# number of rows and columns of data in the training set

dim(training)

# number of rows and columns of data in the validating set

dim(validating)
```

```
## [1] 11776   159
## [1] 7846  159
```
## Data Exploration and Cleaning

Since we choose a random forest model and we have a data set with too many columns, first we check if we have many problems with columns without data. So, remove columns that have less than 60% of data entered.


```r
# Number of cols with less than 60% of data
sum((colSums(!is.na(training[,-ncol(training)])) < 0.6*nrow(training)))
```

[1] 100

```r
# apply our definition of remove columns that most doesn't have data, before its apply to the model.

Keep <- c((colSums(!is.na(training[,-ncol(training)])) >= 0.6*nrow(training)))
training   <-  training[,Keep]
validating <- validating[,Keep]

# number of rows and columns of data in the final training set

dim(training)
```

[1] 11776    59

```r
# number of rows and columns of data in the final validating set

dim(validating)
```

[1] 7846   59

## Modeling
In random forests, there is no need for cross-validation or a separate test set to get an unbiased estimate of the test set error. It is estimated internally, during the execution. So, we proceed with the training the model (Random Forest) with the training data set.


```r
model <- randomForest(classe~.,data=training)
print(model)
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.19%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3348    0    0    0    0 0.000000000
## B    3 2276    0    0    0 0.001316367
## C    0    9 2044    1    0 0.004868549
## D    0    0    5 1924    1 0.003108808
## E    0    0    0    3 2162 0.001385681
```

### Model Evaluate
And proceed with the verification of variable importance measures as produced by random Forest:


```r
importance(model)
```

```
##                      MeanDecreaseGini
## user_name                  90.3730521
## raw_timestamp_part_1      932.6373613
## raw_timestamp_part_2       11.2278639
## cvtd_timestamp           1417.2774858
## new_window                  0.2375014
## num_window                538.0882905
## roll_belt                 547.0536679
## pitch_belt                289.9572495
## yaw_belt                  342.2318442
## total_accel_belt          110.0633448
## gyros_belt_x               39.5889951
## gyros_belt_y               45.1635594
## gyros_belt_z              116.7332576
## accel_belt_x               65.1914831
## accel_belt_y               71.6575192
## accel_belt_z              174.5775863
## magnet_belt_x             109.5946175
## magnet_belt_y             198.2364446
## magnet_belt_z             174.1100246
## roll_arm                  123.8385402
## pitch_arm                  56.7411710
## yaw_arm                    80.6033881
## total_accel_arm            26.3682367
## gyros_arm_x                42.2808067
## gyros_arm_y                41.6757042
## gyros_arm_z                19.4557642
## accel_arm_x                94.6270535
## accel_arm_y                54.4922538
## accel_arm_z                40.7576689
## magnet_arm_x              105.2342845
## magnet_arm_y               79.6373607
## magnet_arm_z               57.7204415
## roll_dumbbell             197.6213608
## pitch_dumbbell             75.0525013
## yaw_dumbbell              104.9213658
## total_accel_dumbbell      112.5343776
## gyros_dumbbell_x           42.7839013
## gyros_dumbbell_y          110.7356305
## gyros_dumbbell_z           25.1911639
## accel_dumbbell_x          126.2760046
## accel_dumbbell_y          183.1386045
## accel_dumbbell_z          140.3221880
## magnet_dumbbell_x         234.3036947
## magnet_dumbbell_y         321.8106105
## magnet_dumbbell_z         299.7706537
## roll_forearm              232.9445408
## pitch_forearm             293.0121796
## yaw_forearm                59.1226542
## total_accel_forearm        33.2545324
## gyros_forearm_x            24.9673052
## gyros_forearm_y            41.4192787
## gyros_forearm_z            26.9075827
## accel_forearm_x           133.6714294
## accel_forearm_y            45.3258310
## accel_forearm_z            96.1075329
## magnet_forearm_x           76.6923241
## magnet_forearm_y           76.9926445
## magnet_forearm_z           97.9443069
```

Now we evaluate our model results through confusion Matrix.


```r
confusionMatrix(predict(model,newdata=validating[,-ncol(validating)]),validating$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    0    0    0    0
##          B    1 1518    5    0    0
##          C    0    0 1362    1    0
##          D    0    0    1 1285    1
##          E    0    0    0    0 1441
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9989          
##                  95% CI : (0.9978, 0.9995)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9985          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   1.0000   0.9956   0.9992   0.9993
## Specificity            1.0000   0.9991   0.9998   0.9997   1.0000
## Pos Pred Value         1.0000   0.9961   0.9993   0.9984   1.0000
## Neg Pred Value         0.9998   1.0000   0.9991   0.9998   0.9998
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1736   0.1638   0.1837
## Detection Prevalence   0.2843   0.1942   0.1737   0.1640   0.1837
## Balanced Accuracy      0.9998   0.9995   0.9977   0.9995   0.9997
```

And confirmed the accuracy at validating data set by calculate it with the formula:


```r
accuracy <-c(as.numeric(predict(model,newdata=validating[,-ncol(validating)])==validating$classe))

accuracy <-sum(accuracy)*100/nrow(validating)
```

Model Accuracy as tested over Validation set = **99.9%**.  

### Model Test

Finally, we proceed with predicting the new values in the testing csv provided, first we apply the same data cleaning operations on it and coerce all columns of testing data set for the same class of previous data set. 

#### Getting Testing Dataset


```r
testingLink <- getURL("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
pml_CSV  <- read.csv(text = testingLink, header=TRUE, sep=",", na.strings=c("NA",""))

pml_CSV <- pml_CSV[,-1] # Remove the first column that represents a ID Row
pml_CSV <- pml_CSV[ , Keep] # Keep the same columns of testing dataset
pml_CSV <- pml_CSV[,-ncol(pml_CSV)] # Remove the problem ID

# Apply the Same Transformations and Coerce Testing Dataset

# Coerce testing dataset to same class and strucuture of training dataset 
testing <- rbind(training[100, -59] , pml_CSV) 
# Apply the ID Row to row.names and 100 for dummy row from testing dataset 
row.names(testing) <- c(100, 1:20)
```

#### Predicting with testing dataset


```r
predictions <- predict(model,newdata=testing[-1,])
print(predictions)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

#### The following function to create the files to answers the Prediction Assignment Submission:


```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0(pathAnswers,"answers/problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(predictions)

#get the time
```


```r
endTime <- Sys.time()
```
The analysis was completed on Sat Oct 19 12:23:09 PM 2015  in 64 seconds.
