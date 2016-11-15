library(dplyr)
library(caret)
library(parallel)
library(doSNOW)

trnData <- read.csv("Data/train.csv")
tstData <- read.csv("Data/test.csv")

labels <- factor(trnData$Survived) 

#Remove unneeded columns
trainSet <- select(trnData,-PassengerId,-Name,-Ticket,-Cabin,-Survived)
testSet <- select(tstData,-PassengerId,-Name,-Ticket,-Cabin)

#Split into training and validation data
trainInd <- createDataPartition(labels, p = 0.8, list=FALSE)
valData <- trainSet[-trainInd,]
valLabels <- labels[-trainInd]
trainData <- trainSet[trainInd,]
trainLabels <- labels[trainInd]

#Train
cl <- makeCluster(4, type = "SOCK")
registerDoSNOW(cl)

#PreProcess
preProc <- c("knnImpute","pca","center","scale")
#Perform cross validation on training
fitControl <- trainControl(method="repeatedcv", repeats = 5, allowParallel = TRUE)

#Gradient Boosting model
gbmFit <- train(trainLabels ~ .,
                data = trainData,
                method = "gbm",
                trControl = fitControl,
                preProcess= preProc,
                verbose = FALSE)
save(gbmFit,file = "gbmFit.Rmodel")

#Random Forest model
rfFit <- train(trainLabels ~ .,
               data = trainData,
               method = "rf",
               trControl = fitControl,
               preProcess= preProc,
               verbose = FALSE)
save(rfFit,file = "rfFit.Rmodel")

#SVM Model
svmFit <- train(trainLabels ~ .,
                data = trainData,
                method= "svmLinear",
                trControl = fitControl,
                preProcess = preProc,
                verbose = FALSE)
save(svmFit,file = "svmFit.Rmodel")

#kNN Model
knnFit <- train(trainLabels ~ .,
                data = trainData,
                method= "knn",
                trControl = fitControl,
                preProcess = preProc,
                verbose = FALSE)
save(knnFit,file = "knnFit.Rmodel")

#kNN Model
lmFit <- train(trainLabels ~ .,
                data=trainData,
                method= "lm",
                trControl = fitControl,
                preProcess = preProc,
                verbose = FALSE)
save(lmFit,file = "knnFit.Rmodel")

stopCluster(cl)

predgbm <- predict(gbmFit,newdata = valData)
predrf <- predict(rfFit,newdata = valData)
predsvm <- predict(svmFit,newdata = valData)

cmgbm <- confusionMatrix(predgbm,valLabels)
cmgbm
cmrf <- confusionMatrix(predrf,valLabels)
cmrf
cmsvm <- confusionMatrix(predsvm,valLabels)
cmsvm


