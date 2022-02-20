read.csv("~/Desktop/Fundamentals of Machine Learning/UniversalBank.csv")

summary(UniversalBank)

#lets get rid of the two variables we don't need
UniversalBank$ID<-NULL
UniversalBank$ZIP.Code<-NULL
#Task 1
UniversalBank$Personal.Loan=as.factor(UniversalBank$Personal.Loan)
summary(UniversalBank)

#Call the libraries
library(lattice)
library(ggplot2)
library(caret)
library(class)

#Normalisation of the data
UniversalBank1_Norm<-UniversalBank

Norm_model<-preProcess(UniversalBank[,-8],method = c("center","scale"))
UniversalBank1_Norm[,-8]=predict(Norm_model, UniversalBank[,-8])
summary(UniversalBank1_Norm)

UniversalBank1_Norm$Personal.Loan=UniversalBank$Personal.Loan


#Train
train.index=createDataPartition(UniversalBank$Personal.Loan,p=0.6, list = FALSE)
train.df=UniversalBank1_Norm[train.index,]
valid.df=UniversalBank1_Norm[-train.index,]


#To predict
To_Predict=data.frame(Age=40, Experience=10, Income=84, Family=2, CCAvg=2, Education_1=0,
                      Mortgage=0, Securities.Account=0, CD.Account=0, Online=1, CreditCard=1)
print(To_Predict)

#Now, we need to apply the normalisation to this record. We must use the same model.
To_Predict_norm=predict(Norm_model,To_Predict)
print(To_Predict_norm)

#Now we will use the Knn function to make the prediction
Prediction <-knn(train = train.df[,1:7],
                 test = To_Predict_norm[,1:7],
                 cl=train.df$Personal.Loan,
                 k=1)
print(Prediction)

#Task 2
#The best choice of K= 3 which prevents the model form over fitting and ignoring the predictor information.
#setseed
#setting the seed of the random number generator will make sure that results are productive
set.seed(123)

fitControl <- trainControl(method= "repeatedcv",
                           number = 3,
                           repeats = 2)

searchGrid=expand.grid(k =1:10)

knn.model=train(Personal.Loan~.,
                data=train.df,
                method='knn',
                tuneGrid = searchGrid,
                trControl = fitControl,)

knn.model

Predictions<-predict(knn.model,valid.df)

#Task 3
#Confusion Matrix
confusionMatrix(Predictions, valid.df$Personal.Loan)

#Task 4
library(class)
#prediction
customer.df= data.frame(Age=40, Experience=10, Income=84, Family=2, CCAvg=2, Education_1=0,
                        Mortgage=0, Securities.Account=0, CD.Account=0, Online=1, CreditCard=1)
knn.4 <- knn(train = train.df[,-8],test = customer.df, cl = train.df[,8], k=3, prob=TRUE)
knn.4

#Task 5
#Repartition the data this time into training, validation and test set.seed()
#50% data for training
#30% data for validation
#20% data for test sets
# K value used = 3

set.seed(1)
train.index <- sample(rownames(UniversalBank1_Norm), 0.5*dim(UniversalBank1_Norm)[1])
set.seed(1)
valid.index <- sample(setdiff(rownames(UniversalBank1_Norm),train.index), 0.3*dim(UniversalBank1_Norm)[1])
test.index = setdiff(rownames(UniversalBank1_Norm), union(train.index, valid.index))


train.df <- UniversalBank1_Norm[train.index,]
valid.df <- UniversalBank1_Norm[valid.index,]
test.df <- UniversalBank1_Norm[test.index,]


norm.values <- preProcess(train.df[, -c(8)], method=c("center", "scale"))
train.df[, -c(8)] <- predict(norm.values, train.df[, -c(8)])
valid.df[, -c(8)] <- predict(norm.values, valid.df[, -c(8)])
test.df[,-c(8)] <- predict(norm.values, test.df[,-c(8)])

testknn <- knn(train = train.df[,-c(8)],test = test.df[,-c(8)], cl = train.df[,8], k=3, prob=TRUE)
validknn <- knn(train = train.df[,-c(8)],test = valid.df[,-c(8)], cl = train.df[,8], k=3, prob=TRUE)
trainknn <- knn(train = train.df[,-c(8)],test = train.df[,-c(8)], cl = train.df[,8], k=3, prob=TRUE)

confusionMatrix(testknn, test.df[,8])
confusionMatrix(validknn, valid.df[,8])
confusionMatrix(trainknn, train.df[,8])

#Test Accuracy: 0.961
#Valid Accuracy: 0.96
#Training Accuracy: 0.9712
#As the model is being fit on the training data it would make sense to say that the classifications are most accurate on the training data set



