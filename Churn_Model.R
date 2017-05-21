#-----------------------------------------------------------------------#

library(xgboost)
library(qlcMatrix)
library(data.table)
library(ggplot2)
library(DiagrammeR)
library(leaps)

#-----------------------------------------------------------------------#

# load data

# read in csv files
churnTrain <- read.csv("D:\\Users\\US52577\\Desktop\\Churn Files\\Data for Models\\R Data\\churnTrain.csv", header = TRUE, strip.white = TRUE)
churnTest <- read.csv("D:\\Users\\US52577\\Desktop\\Churn Files\\Data for Models\\R Data\\churnTest.csv", header = TRUE, strip.white = TRUE)
table(churnTrain$churn)

#-----------------------------------------------------------------------#

# graphical representation of customer breakdown
graph <- ggplot(churnTrain, aes(x=churn, fill = ..count.., col = "red")) + geom_bar()
graph <- graph + ylab("Number of Customers") + xlab("Customer Churn") + labs(title = "Breakdown of Customers by Churn")
graph

#-----------------------------------------------------------------------#

# feature selection

# create feature selection dataset and drop state and area code
data.select <- churnTrain
data.select$state <- NULL
data.select$area_code <- NULL
data.select$churn <- ifelse(data.select$churn == "yes", 1, 0)
data.select$churn <- as.factor(data.select$churn)

# forward stepwise 
regfit.fwd <- regsubsets(churn~. ,data = data.select, nvmax = 15 ,method = "forward")

# model selection criteria 
regsummary.fwd <- summary(regfit.fwd)
which.min(regsummary.fwd$cp) # evidence that 8 variable model is best 

# best model as determined by adjR2
names(coef(regfit.fwd, 8))

#-----------------------------------------------------------------------#

# logistic classification

# build logistic classification model
logistic.model <- glm(churn~total_day_charge+total_intl_calls+international_plan+total_eve_minutes+
                        total_intl_charge+voice_mail_plan+total_night_charge+number_customer_service_calls,
                      data = data.select, family = "binomial")

# summary of model
summary(logistic.model)

#-----------------------------------------------------------------------#

# XGboost training model

# drop state column
churnTrain$state <- NULL

# transform to sparse matrix
sparse_matrix <- sparse.model.matrix(churn ~ .-1, data = churnTrain)

# setting output vector
churnTrain$outputVector = 0
churnTrain$outputVector[churnTrain$churn == "yes"] = 1
outputVector <- churnTrain[, "outputVector"]

# building model
bst <- xgboost(data = sparse_matrix, label = outputVector, max.depth = 10,
               eta = 1, nthread = 2, nround = 5, objective = "binary:logistic")

#-----------------------------------------------------------------------#

# apply trained model to test set

# drop state from test set
churnTest$state <- NULL

# saving test label
testLabel <- churnTest$churn

# transforming test to sparse
sparse_test_matrix <- sparse.model.matrix(churn~.-1, data=churnTest)

# grab label outcome for test vector
churnTest$outputVector = 0
churnTest$outputVector[churnTest$churn == "yes"] = 1
outputTestVector <- churnTest[, "outputVector"]

# making prediction on test data
pred <- predict(bst, sparse_test_matrix)

# changing prediction to binary
prediction <- as.numeric(pred > 0.5)

# determine average model error
err <- mean(as.numeric(pred > 0.5) != outputTestVector)
print(paste("test-error =", err))

#-----------------------------------------------------------------------#

# transforming data into packaged export

# adding in columns for final dataset export
model.probabilities <- data.frame(pred)
model.predictions <- data.frame(prediction)
model.predictions$prediction <- ifelse(model.predictions == 1, "yes", "no")
xgb.final <- cbind(churnTest, model.predictions, model.probabilities)
xgb.final$outputVector <- NULL
xgb.final$churn <- as.character(xgb.final$churn)
xgb.final$matching.prediction <- ifelse(xgb.final$churn == xgb.final$prediction, "match", 
                                        "no match")

# prediction breakdown
xgb.final$predict_breakdown <- ifelse(xgb.final$churn == "yes" & xgb.final$prediction == "yes", "True Positive", ifelse(xgb.final$churn == "yes" & xgb.final$prediction == "no", 
                                                                                                                        "False Negative", ifelse(xgb.final$churn == "no" & xgb.final$prediction == "no", "True Negative", "False Positive")))

# rename columns
setnames(xgb.final, old = c("prediction", "pred", "matching.prediction", "predict_breakdown"), 
         new = c("xgb model prediction", "xgb model probability of churn", "matching prediction", "prediction breakdown"))

# order columns
xgb.final <- xgb.final[,c(19,20,22,23,21,1:18)]

# DF for header display
head.df <- xgb.final[,c("churn","xgb model prediction", "matching prediction", "prediction breakdown", "xgb model probability of churn")]

setnames(head.df, old = c("xgb model prediction", "matching prediction", "prediction breakdown", "xgb model probability of churn"),
         new = c("model pred", "match", "breakdown", "model prob"))

head(head.df)

#-----------------------------------------------------------------------#

# analyzing true positive & true negative predictive accuracy

# breakdown of accurate churn and retention

# set total churn 
churn.number <- sum(xgb.final$churn=="yes")
churn.pred.correct <- sum(xgb.final$`prediction breakdown`=="True Positive")

# xgboost model correctly predicted 
identify.churn <- function(churn.number, churn.pred.correct){
  
  churn.accuracy.rate <<- churn.pred.correct / churn.number
  
  print(sprintf("the model accuracy with respect to accurately predicted churn is %f", churn.accuracy.rate))
  
}

# run function on data
identify.churn(churn.number, churn.pred.correct)

# function for retention accuracy
non.churn <- sum(xgb.final$churn=="no")
non.churn.pred <- sum(xgb.final$`prediction breakdown`=="True Negative")

# xgboost model correctly predicted 
identify.retention <- function(non.churn, non.churn.pred){
  
  retention.accuracy.rate <- non.churn.pred / non.churn
  
  print(sprintf("the model accuracy with respect to accurately predicted retention is %f", retention.accuracy.rate))
  
}

# run function on data
identify.retention(non.churn, non.churn.pred)

#-----------------------------------------------------------------------#

# graphing accuracy rates

# visualizing relative accuracy rates
accuracy.data <- data.frame(`Churn Category` = c("Retained", "Not Retained"), 
                            `Predictive Accuracy` = c(.99, .71))

accuracy.data$Churn.Category <- as.character(accuracy.data$Churn.Category)
accuracy.graph <- ggplot(accuracy.data, aes(x=Churn.Category, y=Predictive.Accuracy, fill = Churn.Category)) + geom_bar(stat = "identity")
accuracy.graph <- accuracy.graph + ylab("Predictive Accuracy") + xlab("Customer Class") + labs(title = "Predictive Accuracy with Resepct to Customer Class")
accuracy.graph

#-----------------------------------------------------------------------#

# feature importance

# generating importance matrix
importance_matrix <-  xgb.importance(feature_names = sparse_matrix@Dimnames[[2]], model = bst)
head(importance_matrix)

# generating plot that shows importance
xgb.ggplot.importance(importance_matrix = importance_matrix)

#-----------------------------------------------------------------------#

# validate results with 10 fold CV

# validate results with CV
bst.CV <- xgb.cv(data = sparse_matrix, label = outputVector, max.depth = c(15),
                 eta = 1, nthread = 2, nround = 5, nfold = 10, objective = "binary:logistic",
                 prediction = TRUE)

#-----------------------------------------------------------------------#

# run model on hypothetical customers

# predict new customer using best trained model
new.customer <- data.frame(account_length = c(100, 98), area_code = c("area_code_415", "area_code_408"), international_plan = c("yes", "no"), voice_mail_plan = c("yes", "no"), 
                           number_vmail_messages = c(20, 25), total_day_minutes=c(200, 195), total_day_calls=c(100, 95), total_day_charge=c(40, 45),
                           total_eve_minutes=c(200, 180), total_eve_calls=c(100, 90), total_eve_charge=c(20, 25), total_night_minutes=c(200, 190),
                           total_night_calls=c(100, 80), total_night_charge=c(10, 8), total_intl_minutes=c(15, 10), total_intl_calls=c(3, 2),
                           total_intl_charge=c(3, 1), number_customer_service_calls=c(2, 5))

# sparse matrix conversion
sparse_matrix_pred <- sparse.model.matrix(~.-1, data=new.customer)

# making prediction
pred_new_data <- predict(bst, sparse_matrix_pred)

# changing prediction to binary
prediction_pred <- as.numeric(pred_new_data > 0.5)

# creating data.frame for new predictions
final_results <- data.frame(new.customer, prediction_pred, pred_new_data)
final_results$prediction_pred <- ifelse(prediction_pred==0, "no", "yes")
final_results$pred_new_data
final_results$prediction_pred

# end of script