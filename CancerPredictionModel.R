library(caret)
library(corrplot)
library(ggplot2)
library(rpart)
library(kknn)
library(hydroGOF)
library(cowplot)
library(caret)
library(randomForest)
library(e1071)

originalCancerData = read.csv("cData.csv")

originalCancerData= originalCancerData[,c(-1,-2,-34)]#A lot of NA's in both coloumns 2 and 34. id has low correlation to the diagnosis which is why we dropped the feature.
ggplot(data1, aes(diagnosis)) + geom_bar(fill = "red")
#we observe data points with no diagnosis, and with several NA's.
cleanedUpCancerData = originalCancerData[complete.cases(originalCancerData),]
#Checking for NA's.
cleanedUpCancerData[is.na(cleanedUpCancerData)]
cleanedUpCancerData = cleanedUpCancerData[-which(cleanedUpCancerData$diagnosis==""),]
#data[1,1] = "myValue"
str(originalCancerData)

#Plot One Variable Graph.
data$diagnosis = as.factor[]

table(cleanedUpCancerData$diagnosis)
prop.table(table(cleanedUpCancerData$diagnosis))

#Converting Coloumn One to Numeric then drawing a correlation plot.
cleanedUpCancerData$diagnosis=ifelse(cleanedUpCancerData$diagnosis == "M",1,0)
#Using the 
cancerCorrelationPlot = cor(cleanedUpCancerData)
cancerCorrelationPlot
corrplot(cancerCorrelationPlot, method = "color", type = "lower")

#Selecting just the Mean,SE, and Worst parts of the Cancer Data and plotting them.
correlationMeanCancerData = cor(cleanedUpCancerData[,c(1:11)])
correlationSECancerData = cor(cleanedUpCancerData[,c(12:21)])
correlationWorstCancerData = cor(cleanedUpCancerData[,c(22:31)])
corrplot(correlationMeanCancerData, method = "color", type = "lower" )
corrplot(correlationSECancerData, method = "color", type = "lower" )
corrplot(correlationWorstCancerData, method = "color", type = "lower" )

#Don't use the same variables if they have the same correlation.

#heatmap.
heatmap(cleanedUpCancerData,col=col,symm=F)

#Creating a Cleaned Up Cancer Data Reference with the NA's and blank spaces removed, but diagnosis still in the character data type so the boxplot can recognize the fill.
cleanedUpCancerDataForVisualization = originalCancerData[,c(-2,-34)]
cleanedUpCancerDataForVisualization = cleanedUpCancerDataForVisualization[complete.cases(cleanedUpCancerDataForVisualization),]
cleanedUpCancerDataForVisualization[is.na(cleanedUpCancerDataForVisualization)]
cleanedUpCancerDataForVisualization = cleanedUpCancerDataForVisualization[-which(cleanedUpCancerDataForVisualization$diagnosis==""),]

p1=ggplot(cleanedUpCancerDataForVisualization, aes(y=radius_mean, fill=diagnosis)) + geom_boxplot()
p2=ggplot(cleanedUpCancerDataForVisualization, aes(y=texture_mean, fill=diagnosis)) + geom_boxplot()
p3=ggplot(cleanedUpCancerDataForVisualization, aes(y=perimeter_mean, fill=diagnosis)) + geom_boxplot()
p4=ggplot(cleanedUpCancerDataForVisualization, aes(y=area_mean, fill=diagnosis)) + geom_boxplot()
p5=ggplot(cleanedUpCancerDataForVisualization, aes(y=smoothness_mean, fill=diagnosis)) + geom_boxplot()
p6=ggplot(cleanedUpCancerDataForVisualization, aes(y=compactness_mean, fill=diagnosis)) + geom_boxplot()
p7=ggplot(cleanedUpCancerDataForVisualization, aes(y=concavity_mean, fill=diagnosis)) + geom_boxplot()
p8=ggplot(cleanedUpCancerDataForVisualization, aes(y=concave.points_mean, fill=diagnosis)) + geom_boxplot()
p9=ggplot(cleanedUpCancerDataForVisualization, aes(y=symmetry_mean, fill=diagnosis)) + geom_boxplot()
p10=ggplot(cleanedUpCancerDataForVisualization, aes(y=fractal_dimension_mean, fill=diagnosis)) + geom_boxplot()

plot_grid(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10, nrow=3)
###########Visulization Complete.

#####Split into Test and Train Data.



#Splitting Test and Training Data.
set.seed(17)
holderSplit = createDataPartition(cleanedUpCancerData$diagnosis,p=0.7,list = F)
trainingData = cleanedUpCancerData[holderSplit,]
testingData = cleanedUpCancerData[-holderSplit,]
#Removing collinearity by picking specific features, and creating a DataSet.

specificCancerFeatures = c('diagnosis', 'texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean')

trainingData.SelectedFeatures = cleanedUpCancerData[holderSplit,specificCancerFeatures]

#Check Imbalances in the Data.
table(trainingData$diagnosis)




#Build model using logistic regression.
cancerModel.lg = glm(diagnosis ~. , data= trainingData.SelectedFeatures, family = binomial(link="logit"))
cancerPredictions = predict(cancerModel.lg, testingData, type="response")
testingData$cancerPredictions=ifelse(cancerPredictions > 0.5, 1, 0)
table(testingData$diagnosis)
#Testing the accuracy for the logistic regression model.
accuracyCancerPredictions = mean(testingData$cancerPredictions == testingData$diagnosis)
accuracyCancerPredictions
summary(cancerModel.lg)
cancerPredictions

#Confusion Matrix For logisitic regression model.
cm_lg = confusionMatrix(as.factor(testingData$cancerPredictions), as.factor(testingData$diagnosis))   
cm_lg

#Building Model using KKNN.
table(trainingData$diagnosis)
class(trainingData$diagnosis)
table(testingData$diagnosis)
trainingData$diagnosis = as.factor(trainingData$diagnosis)
cancerModel.kknn = train(diagnosis~., trainingData, method="kknn")
 
accuracyKKNN = mean (predictedCancerModelKKNN == testingData$diagnosis)
accuracyKKNN #accuracy = 97.05882 %

#Build confusion matrix for KKNN model.
cm_KKNN = confusionMatrix(as.factor(predictedCancerModelKKNN), as.factor(testingData$diagnosis))
cm_KKNN

#Build model using Random Forest.
cancerModel.rf = train(diagnosis~., trainingData, method="rf")
predictedCancerModelRF = predict(cancerModel.rf, testingData, method = "rf")

accuracyRF = mean (predictedCancerModelRF == testingData$diagnosis)
accuracyRF#Accuracy is 94.70588 %.

#Build confusion matric for Random Forest.
cm_rf = confusionMatrix(as.factor(predictedCancerModelRF), as.factor(testingData$diagnosis))
cm_rf

#Building the Support Vector Machines.
cancerModel.svm = svm(diagnosis~., data=trainingData)
cancerPredictions.svm=predict(cancerModel.svm, testingData)
accuracy.svm = mean(cancerPredictions.svm==testingData$diagnosis)
accuracy.svm #0.9823529

#Building the Confusion Matrix for the SVM Model.
cm_svm=confusionMatrix(as.factor(cancerPredictions.svm), as.factor(testingData$diagnosis))   
cm_svm


#Build the Rpart Model.
cancerModel.rpart = train(diagnosis~., trainingData, method="rpart")
cancerPredictions.rpart = predict(cancerModel.rpart, testingData)
accuracy.rpart = mean(cancerPredictions.rpart==testingData$diagnosis)
accuracy.rpart #0.9470588

#Build the Confusion Matrix for the Rpart Model.
confusionMatrix_rpart = confusionMatrix(as.factor(cancerPredictions.rpart), as.factor(testingData$diagnosis))   
confusionMatrix_rpart


#Build Validation Table.colnames = diagnosis, diagnosis.lg, diagnosis.knn.
modelsUsedColoumnNameVTable = c("Diagnosis","Logistic Regression", "Random Forest", "KKNN", "SVM", "R-Part")
modelsUsedVTable = c(testingData$diagnosis,cancerPredictions,predictedCancerModelKKNN,predictedCancerModelRF,cancerPredictions.svm,cancerPredictions.rpart)
cancerPredictionsVTable = data.frame(modelsUsedVTable, modelsUsedColoumnNameVTable)

#Build an Observation Table.
modelsUsed = c("Logistic Regression", "Random Forest", "KKNN", "SVM", "R-Part")
accuracy = c(accuracyCancerPredictions, accuracyRF, accuracyKKNN, accuracy.svm, accuracy.rpart)
observationTable = data.frame(modelsUsed, accuracy)


#Do Feature Engineering.
cancerModel.lg = glm(diagnosis ~. , data= trainingData, family = binomial(link="logit"))
cancerPredictions = predict(cancerModel.lg, testingData, type="response")
accuracy1 = mean(cancerPredictions == testingData$diagnosis)
accuracy1
cm_lg = confusionMatrix(as.factor(cancerPredictions), as.factor(testingData$diagnosis))   
cm_lg
summary(cancerModel.lg)

#creating a mean features.

#Do feature engineering for KKNN Model. 
library(hydroGOF)

cancerMeanSelectedFeatures = c("diagnosis","radius_mean")
cancerMeanPossibleFeatures = c("texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave.points_mean", "symmetry_mean","fractal_dimension_mean")
training = cleanedUpCancerData[seq(1,nrow(cleanedUpCancerData),2), cancerMeanSelectedFeatures]
testing = cleanedUpCancerData[seq(2,nrow(cleanedUpCancerData),2),cancerMeanSelectedFeatures]
featureEngineeringModel = train(diagnosis~., training, method="kknn")
predictedFeatureEngineering = predict(featureEngineeringModel, testing)
baseAccuracy = mean(predictedCancerModelKKNN == testingData$diagnosis)
baseAccuracy
cleanedUpCancerData$diagnosis = as.factor(cleanedUpCancerData$diagnosis)
for (i in 1: length(cancerMeanPossibleFeatures)){
print(i)
  training = cleanedUpCancerData[seq(1,nrow(cleanedUpCancerData),2), c(cancerMeanSelectedFeatures,cancerMeanPossibleFeatures[i])]
  testing = cleanedUpCancerData[seq(2,nrow(cleanedUpCancerData),2), c(cancerMeanSelectedFeatures,cancerMeanPossibleFeatures[i])]
  featureEngineeringModel= train(diagnosis~., training, method = "kknn")
  predictedFeatureEngineering = predict(featureEngineeringModel, testing)
  newAccuracyKKNN = mean(predictedCancerModelKKNN == testingData$diagnosis)
  newAccuracyKKNN
  
  
  if (newAccuracyKKNN>accuracyKKNN){
    print(cancerMeanSelectedFeatures)
    cancerMeanSelectedFeatures = c(cancerMeanSelectedFeatures, cancerMeanPossibleFeatures[i])
    newAccuracyKKNN = accuracyKKNN
    print(i)
    paste("Features Selected are", cancerMeanSelectedFeatures)
  }
}


