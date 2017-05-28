#---------------------------------------------------------------#

library(neuralnet)
library(caTools)
library(mlbench)
library(caret)

#---------------------------------------------------------------#

# load data

# read in csv files
churnTrain <- read.csv("D:\\Users\\US52577\\Desktop\\Churn Files\\Data for Models\\R Data\\churnTrain.csv", header = TRUE, strip.white = TRUE)
churnTest <- read.csv("D:\\Users\\US52577\\Desktop\\Churn Files\\Data for Models\\R Data\\churnTest.csv", header = TRUE, strip.white = TRUE)

#---------------------------------------------------------------#

# combine data
churn <- rbind(churnTrain, churnTest)
rm(churnTest, churnTrain)

# reorder columns 
churn <- churn[c(20,1,3,4,5,2,6:19)]


# Create Vector of Column Max and Min Values
maxs <- apply(churn[,6:20], 2, max)
mins <- apply(churn[,6:20], 2, min)
maxs
mins

# Use scale() and convert the resulting matrix to a data frame
scaled.data <- as.data.frame(scale(churn[,6:20], center = mins, scale = maxs - mins))

#---------------------------------------------------------------#

# Convert Private column from Yes/No to 1/0
churn.col <- as.numeric(churn$churn)-1
data <- cbind(churn.col, scaled.data)

# set random seed
set.seed(101)

# Create Split - you can choose any column 
split <- sample.split(data$churn.col, SplitRatio = 0.70)

# Split based off of split Boolean Vector
train <- subset(data, split == TRUE)
test <- subset(data, split == FALSE)

#---------------------------------------------------------------#

# grab names for variables
features <- names(scaled.data)

# Concatenate strings
f <- paste(features,collapse = ' + ')
f <- paste('churn.col ~' ,f)

# Convert to formula
f <- as.formula(f)

# review form - this is how R takes output and input variables with most models
f

#---------------------------------------------------------------#

# build the neural network using training data
nn <- neuralnet(f, train, hidden=c(5, 5, 5), linear.output=FALSE)

# Compute Predictions off Test Set
predicted.nn.values <- compute(nn, test[2:16])

# create data frame of node outputs and isolate individual probabilities
probs <- data.frame(predicted.nn.values)
probs$pred <- ifelse(probs$net.result > .5, 1, 0)

# grab prediction columns
probabilities <- probs[, 35:36]

# cbind columns to test data
output.data <- cbind(test, probabilities)

# create match and prediction breakdown columns
output.data$match <- ifelse(output.data$churn.col == output.data$pred, "yes", "no")
output.data$`prediction breakdown` <- ifelse(output.data$churn.col == 1 & output.data$pred == 1, "true positive",
                                             ifelse(output.data$churn.col == 1 & output.data$pred == 0, "false negative",
                                                    ifelse(output.data$churn.col == 0 & output.data$pred == 1, "false positive",
                                                           "true negative")))

# overall model accuracy and granular view - same as prediction breakdown column
model.accuracy <- sum(output.data$match == "yes") / nrow(output.data)
conf.mat <- confusionMatrix(test$churn.col, output.data$pred)
conf.mat

#---------------------------------------------------------------#

# visualize neural network 
plot(nn)

# end Neural Network Script