library(caret)
library(kernlab)
library(mlbench)
library(gridExtra)
library(rpart)
library(rpart.plot)
library(corrplot)
#loading the data
trainingRaw<-read.csv("~/Documents/repos/MachineLearning/pml-training.csv", na.strings = c("#DIV/0!","",NA), sep=",")
testingRaw<-read.csv("~/Documents/repos/MachineLearning/pml-testing.csv", na.strings = c("#DIV/0!","",NA), sep=",")
head(training)
unique(training$classe)

#cleaning the data (removing the X variable)
training<-trainingRaw[,-1]
testing<-testingRaw[,-1]

#staying only with numeric variables
training<-training[,sapply(training, is.numeric)]
testing<-testing[,sapply(testing, is.numeric)]

#removing NA
#all the columns which the sum is 0, has no NA
removeNA=apply(apply(training, FUN=is.na, 2), FUN=sum, 2)
training<-training[,removeNA==0]
training$classe<-trainingRaw$classe
dim(training)

removeNA=apply(apply(testing, FUN=is.na, 2), FUN=sum, 2)
testing<-testing[,removeNA==0]
dim(testing)

#creating cross-validation
set.seed(123)
inTrain = createDataPartition(training$classe, p= 0.6, list=FALSE)
cross_training=training[inTrain,]
cross_testing=training[-inTrain,]

#since it is a classification problem we will use trees, because they are very little influenced by outliers,
#distribution
#lets test which tree model to use

control <- trainControl(method="repeatedcv", repeats=3, number=5)
#Bagging
set.seed(123)
Bagging<- train(classe~., data=cross_training, method="treebag",trControl=control, verbose=FALSE)
Bagging
#boosted Trees
set.seed(123)
Boosted <- train(classe~., data=cross_training, method="gbm", trControl=control, verbose=FALSE)
Boosted
#Random Forest
set.seed(123)
RandFor<- train(classe~., data=cross_training, method="rf", trControl=control, verbose=FALSE)
RandFor

#Just to be sure, lets check non-tree method accuracy
set.seed(123)
SVM<-train(classe~., data=cross_training, method="svmRadial", trControl=control, verbose=FALSE)
SVM


#comparing methods
results<-resamples(list(Bagging = Bagging, Boosted = Boosted, RandFor = RandFor, SVM = SVM))
summary(results)
g1<-bwplot(results)
g2<-dotplot(results)
grid.arrange(g1,g2,ncol=1)

#testing the best model
prediction<-predict(RandFor, cross_testing)
CMatrix<-confusionMatrix(prediction, cross_testing$classe)
C.data.frame<-as.data.frame(CMatrix$table)

#figures
corrplot(cor(cross_training[, -length(cross_training)]), method="color")
rpart(RandFor$finalModel)
g3<-ggplot(C.data.frame)+geom_tile(aes(x=Reference,y=Prediction,fill=Freq))
g3
