#This is my first machine learning project. I have started this after going through the other solutions to get an inspiration. I am a beginner and feedback is much appreciated. 
library(ggplot2)
library(dplyr)
library(Amelia)
library(pscl)
library(ROCR)
library(caret)

#Reading training data and test data from CSV files
train.data<-read.csv('train.csv', header=T, stringsAsFactors=F)
test.data<-read.csv('test.csv', header=T, stringsAsFactors=F)

#combining both train.data and test.data using dplyr package
titanic.data<-bind_rows(train.data, test.data)
#This gives no of observations and no of variables in the data set
dim(titanic.data)
#This gives the names of the variables in the data set
names(titanic.data)

#Lets do some feature engineering and get more meaningful data
#One thing that strikes me is the Name variable. Lets extract the title from the name and group them
#Ex:titanic.data$Name[1] -- "Braund, Mr. Owen Harris"
titanic.data$Title<-gsub('(.*, )|(\\..*)','',titanic.data$Name)
unique(titanic.data$Title)
table(titanic.data$Sex, titanic.data$Title)

#Grouping titles of similar meaning to similar groups. gsub checks for the pattern, replaces with the given pattern in a dataset.
titanic.data$Title<-gsub('Mlle|Ms', 'Miss', titanic.data$Title)
titanic.data$Title<-gsub('Mme', 'Mrs', titanic.data$Title)
titanic.data$Title<-gsub('Don|Rev|Dr|Major|Lady|Sir|Col|Capt|the Countess|Jonkheer|Dona','Other', titanic.data$Title)
unique(titanic.data$Title)
table(titanic.data$Sex, titanic.data$Title)

#We can create another variable called FamilySize using Sibsp and Parch as those reveal info about his/her family and see their survival.
titanic.data$FamilySize=titanic.data$SibSp+titanic.data$Parch+1
#Gives the unique values of the FamilySize variable
unique(titanic.data$FamilySize)
qplot(factor(titanic.data$FamilySize[1:891]),data=titanic.data[1:891,], geom="bar",fill=factor(titanic.data$Survived[1:891]), xlab='Size of the family', ylab=' Frequency', main='Size of the family and its survival status')

#Grouping the families according to their sizes
titanic.data$FSize[titanic.data$FamilySize==1]<-'singletion'
titanic.data$FSize[titanic.data$FamilySize>1 & titanic.data$FamilySize<=5]<-'small'
titanic.data$FSize[titanic.data$FamilySize>5]<-'large'
unique(titanic.data$FSize)

#Converting variables to factors as required for logistic regression
var_to_factor<-c('PassengerId','Survived','Pclass', 'Sex', 'Embarked', 'Title', 'FSize')
#We need to use lapply, as the mode of the data set is a list.
mode(titanic.data)
titanic.data[var_to_factor]<-lapply(titanic.data[var_to_factor], function(x) as.factor(x))
sapply(titanic.data, function(x) is.factor(x))

#Missing values
missmap(titanic.data, main='Missing values Vs Observed values')
sapply(titanic.data, function(x) sum(is.na(x)))
#Here Age has 263 missing values, Fare has 1 missing values. We ignore the missing values of Survived as they belong to test data

#Let us first handle missing value of Fare.
titanic.data[is.na(titanic.data$Fare),]
#Missing value is found at row 1044
#The passenger started from 'S'and travelled in class '3'. Lets see what others in the same category paid.
ggplot(titanic.data[titanic.data$Pclass == '3' & titanic.data$Embarked == 'S', ],aes(x = Fare)) + geom_density(titanic.data = '#77d6ff',alpha=0.2) + geom_vline(aes(xintercept=median(Fare, na.rm=T)),colour='red', linetype='dotted', lwd=1)

boxplot(titanic.data[titanic.data$Pclass == '3' & titanic.data$Embarked == 'S', ]$Fare, main='Fare')
summary(titanic.data[titanic.data$Pclass == '3' & titanic.data$Embarked == 'S', ]$Fare)
#It is safe to impute with median of values in that category.
titanic.data$Fare[1044]=median(titanic.data[titanic.data$Pclass == '3' & titanic.data$Embarked == 'S', ]$Fare, na.rm=T)
sum(is.na(titanic.data$Fare))

#Lets now handle missing values for Age. There are 263 missing values. We shall impute with the mean value(It is not a good approach as there are many missing values, distribution also changes. Predictive imputation is preferred.)
summary(titanic.data$Age)
boxplot(titanic.data$Age, main='Age in Years')
titanic.data$Age[is.na(titanic.data$Age)]<-mean(titanic.data$Age, na.rm=T)
sum(is.na(titanic.data$Age))
missmap(titanic.data)
hist(titanic.data$Age, freq=F, main='Distribution of Age after imputation', col='grey', ylim=c(0,0.04))

#we can remove cabin variable as it doesn't contain many values
titanic.data$Cabin<-NULL

#Model building and prediction 

#Creating training and testing data sets
train.data<-titanic.data[1:800,]
test.data<-titanic.data[801:891,]
#Lets first add all the predictors
glm.fit1=glm(Survived~ Pclass+Sex+Age+SibSp+Parch+Fare+Embarked+Title+FSize, family=binomial("logit"), data=train.data)
summary(glm.fit1)
anova(glm.fit1, test='Chisq')
#We can remove Fare, Embarked, FSize variables as they have large p-value and adding each of these variables doesn't add any power to model(Residual deviance does not decrease much)
glm.fit1=glm(Survived~ Pclass+Sex+Age+SibSp+Title, family=binomial("logit"), data=train.data)
coef(glm.fit1)
#McFadden’s R2 value is 0.3698508-- which is reasonably good as it is in the range of 0.2 to 0.4
pR2(glm.fit1)
#We can also check the importance of the variables relative o each other using caret package.s
varImp(glm.fit1)
#Predicting the results on the new data set. Got the accuracy of 84.6%, which is awesome
glm.prob<- predict(glm.fit1, newdata=subset(test.data, select=c(3,5,6,7,12)), type='response')
glm.pred<- ifelse(glm.prob>0.5,1,0)
glm.missclass<-mean(glm.pred!=test.data$Survived)
glm.accuracy=1-glm.missclass
#Plotting the ROC curve and calculating the AUC. The corresponding auc value is 0.90, which is really great
glm.pred=prediction(glm.prob, test.data$Survived)
glm.performance=performance(glm.pred, measure = "tpr", x.measure = "fpr")
plot(glm.performance)
glm.auc=performance(glm.pred, measure='auc')
glm.auc=glm.auc@y.values[[1]]


#In the above method, we have splitted the data manually. Hence there is only one statistic. To understand the variability, lets use k-fold cross validation using package 'caret'.
glm.fit2=train(Survived~ Pclass+Sex+Age+SibSp+Title, family='binomial', method='glm',data=train.data, trControl=glm.ctrl, tuneLength=5)
glm.pred2=predict(glm.fit2, newdata=subset(test.data, select=c(3,5,6,7,12)))
confusionMatrix(glm.pred2, test.data$Survived)
#we got an accuracy of 84.62% which is same as we got from manually splitting the data. Hence we can stop here.