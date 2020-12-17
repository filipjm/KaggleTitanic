## Titanic Survival Prediciton 
## Machine Learning in R

# By Phil Juricev


# set up the working directory / path

getwd()
#setwd()


# Import and read the two datasets - train and test
dataTrain <- read.csv("train.csv")
dataTest <- read.csv("test.csv")

## NOTE: My thinking here is that this is a statistical classification problem
## as we are grouping the passengers in either Survived or Not Survived
## Regression fails here -> assumptions: linearity, homoscedasticity, independence, normality
## This is why I would like to use the Random Forest approach for this problem

# import relevant libraries
#install.packages("randomForest")
library(randomForest)

# probably not the best way of data cleaning
# initially I thought of simply following the GIGO (garbage in garbage out) principle
# however, Kaggle wants a certain size to the result file, so we will replace the NA
# values with the mean

#age
dataTrain[is.na(dataTrain$Age),"Age"] <- mean(dataTrain$Age,na.rm = TRUE)
dataTest[is.na(dataTest$Age),"Age"] <- mean(dataTest$Age,na.rm = TRUE)

#fare
dataTrain[is.na(dataTrain$Fare),"Fare"] <- mean(dataTrain$Fare,na.rm = TRUE)
dataTest[is.na(dataTest$Fare),"Fare"] <- mean(dataTest$Fare,na.rm = TRUE)

#could further clean it with regards to Parch or SibSp, etc

## NOTE: What could have impacted the survival of a person? 
## Most certainly Age and Gender -> women and children first, in evacuation, then PClass and Fare
## since in those times the rich had the priority over the poor
## also could include sibsp and parch since the fact of having family on board could have played a role
## I am excluding the point of embarkation - reason: it was not one of the factors giving priority in evacuation
## it could potentially be argued that people from some of the embarkation points were richer on average
## and as a consequence could afford a higher ticket, but that is taken into account already in PClass and Fare

# cast some of the potential predictors into factors

dataTrain$Sex <-as.factor(dataTrain$Sex)
dataTrain$Pclass <-as.factor(dataTrain$Pclass)

# class survival into 0 and 1
dataTrain$Survived <- as.factor(dataTrain$Survived)

dataTest$Sex <-as.factor(dataTest$Sex)
dataTest$Pclass <-as.factor(dataTest$Pclass)

# let's write out the model
modelString <- as.formula("Survived ~ Pclass + Age + Sex + SibSp + Parch + Fare")

# run the Random Forest Algorithm

titanicModel <- randomForest(formula = modelString, data = dataTrain, ntree = 500, mtry = ceiling(sqrt(7)), nodesize = 0.01*length(dataTrain$PassengerId))

# ASIDE: ceiling(sqrt(7)) is chosen as the nearest integer value to the sqrt of the number of predictors

# use that model to run a prediction

Survived <- predict(titanicModel,newdata = dataTest)

# might be useful to check the split of survival

table(Survived)

#let's format the data into a dataframe ready for submission

PassengerId <- dataTest$PassengerId

outputDF <- as.data.frame(PassengerId)
outputDF$Survived <- Survived

# finally submit the outputDF file to Kaggle

write.csv(outputDF,file="titanicPrediction.csv",row.names = FALSE)

# model accuracy: 0.77272






