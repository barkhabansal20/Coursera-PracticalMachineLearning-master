Practical Machine Learning Project Writeup

Following is the analysis for the prediction assignment of the practical machine learning course. The report is kept short and crisp hence the best predictor is shown in the report, other predictors were not as good as the following results hence not included in the report.

Start with loading packages. (caret, randomForest, Hmisc, foreach & doParallel) 

Set the seed value for reproducibility.

```{r}
options(warn=-1)
library(caret)
library(randomForest)
library(Hmisc)

library(foreach)
library(doParallel)
set.seed(4356)
```

Load the csv file data to dataframe and analyze it :

```{r}
data <- read.csv("/projects/Coursera-PracticalMachineLearning/data//pml-training.csv")
#summary(data)
#describe(data)
#sapply(data, class)
#str(data)
```

Two problems observed are :
 1 - Some data have characters ("#DIV/0!")
 2 - Some columns have a lot of missing data/value
 
To manage the first problem, reimport data ignoring "#DIV/0!" values :

```{r}
data <- read.csv("/projects/Coursera-PracticalMachineLearning/data//pml-training.csv", na.strings=c("#DIV/0!") )
```

And force the cast to numeric values for the specified columns (i.e.: 8 to end) :

```{r}
cData <- data
for(i in c(8:ncol(cData)-1)) {cData[,i] = as.numeric(as.character(cData[,i]))}
```

For second problem, select the column with a 100% completion rate. Also remove less important variables as shown below ( "X"", timestamps, "new_window", "num_window" etc). 

```{r}
featuresnames <- colnames(cData[colSums(is.na(cData)) == 0])[-(1:7)]
features <- cData[featuresnames]
```


Split the dataset in two part for training and testing.

```{r}
xdata <- createDataPartition(y=features$classe, p=3/4, list=FALSE )
training <- features[xdata,]
testing <- features[-xdata,]
```


Train a classifier with the training data. Do parallelised processing with the foreach and doParallel package and process 4 random forest with 150 trees each and combine then to have a random forest model with a total of 600 trees.
```{r}
registerDoParallel()
model <- foreach(ntree=rep(150, 4), .combine=randomForest::combine) %dopar% randomForest(training[-ncol(training)], training$classe, ntree=ntree)
```

Evaluate the model with the confusionmatrix method (and check accuracy, sensitivity & specificity metrics) :
```{r}
predictionsTr <- predict(model, newdata=training)
confusionMatrix(predictionsTr,training$classe)


predictionsTe <- predict(model, newdata=testing)
confusionMatrix(predictionsTe,testing$classe)
```

The confusionmatrix shows that the model is good and efficient because it has an accuracy of 0.997 and very good sensitivity & specificity values on the testing dataset. 
Also it is 100% (20/20) correct for the Course Project Submission.

