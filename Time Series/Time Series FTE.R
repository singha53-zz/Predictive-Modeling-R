#----------------------------------------------#

# devtools::install_github("twitter/AnomalyDetection") # AnomalyDetection
library(AnomalyDetection)
library(ggplot2)
library(Rcpp)
library(timeDate)
library(data.table)
library(tseries)
library(lubridate)
library(forecast)
# devtools::install_github("ellisp/forecastxgb-r-package/pkg") # forecastxgb
library(forecastxgb)
library(caret)
library(qlcMatrix)
library(xgboost)

#----------------------------------------------#

# load data
series <- read.csv("Time Series.csv", header = TRUE, strip.white = TRUE)
series$EventDate <- as.character(series$EventDate)
series$EventDate <- mdy(series$EventDate)
setnames(series, old = c("FTE.per.Day"), new = c("FTE Requirement"))

#----------------------------------------------#

# creating dates vector
dates <- seq(as.Date("2016-04-01", format="%Y-%m-%d"), as.Date("2017-03-31", format="%Y-%m-%d"),"days")

# isolate weekdays
weekdays.dates <- dates[ ! weekdays(dates) %in% c("Saturday", "Sunday")]  
dates <- data.frame(as.character(weekdays.dates))
setnames(dates, old = c("as.character.weekdays.dates."), new = "Date")

# convert Event Date column for merge
series$EventDate <- as.character(series$EventDate)
setnames(series, old = c("EventDate"), new = "Date")

# merge for weekday only view
weekdays <- merge(series, dates, by = c("Date"), all.y = TRUE)
series.weekdays <- series
series.weekdays <- na.omit(weekdays)

# convert date field to date
series.weekdays$Date <- as.Date(series.weekdays$Date)

# visualzie trend in FTE Requirement per day
ggplot(series.weekdays, aes(x=Date, y=`FTE Requirement`, color=`FTE Requirement`)) + geom_line() + geom_smooth(method = "lm")+ylab('FTE Requirement')

# convert date for anomaly detection
series.weekdays$Date <- as.POSIXct(series.weekdays$Date)

#----------------------------------------------#

# create anomaly detection subset
data.anomaly <- series.weekdays[,c("Date","FTE Requirement")]

# Apply anomaly detection
data_anomaly <- AnomalyDetectionTs(data.anomaly, max_anoms=0.01, direction="both", 
                                   plot=TRUE, e_value = T)

# visualize anomalies
data_anomaly$plot

#----------------------------------------------#

# forecasting using all observations

# plot all observations
series$Date <- as.Date(series$Date)
ggplot(series, aes(x=Date, y=`FTE Requirement`, color=`FTE Requirement`)) + geom_line() + geom_smooth(method = "lm")+ylab('FTE Requirement')
# it appears little time is being entered on the weekends

# drop outliers in terms of FTE Requirement
series.weekdays <- series.weekdays[!(series.weekdays$`FTE Requirement` < 5),]

# create time series object 
count_ts <- ts(series.weekdays[, c('FTE Requirement')])
series.weekdays$clean_cnt <- tsclean(count_ts)
# in this case, the cleaned FTE Requirement count did not remove outliers

#----------------------------------------------#

# adding ARIMA (auto-regressive integrated moving average)

# calculating weekly and monthly averages
series.weekdays$cnt_ma <- ma(series.weekdays$clean_cnt, order=7) # using the clean count with no outliers
series.weekdays$cnt_ma30 <- ma(series.weekdays$clean_cnt, order=30)
mean(series.weekdays$cnt_ma, na.rm = TRUE)

# plotting with weekly and monthly averages included
ggplot() +
  geom_line(data = series.weekdays, aes(x = Date, y = clean_cnt, colour = "Counts")) +
  geom_line(data = series.weekdays, aes(x = Date, y = cnt_ma,   colour = "Weekly Moving Average"))  +
  geom_line(data = series.weekdays, aes(x = Date, y = cnt_ma30, colour = "Monthly Moving Average"))  +
  ylab('Cases Entered')

# plotting weekly and monthly averages
ggplot() +
  geom_line(data = series.weekdays, aes(x = Date, y = cnt_ma,   colour = "Weekly Moving Average"))  +
  geom_line(data = series.weekdays, aes(x = Date, y = cnt_ma30, colour = "Monthly Moving Average"))  +
  ylab('Cases Entered')

# decomposition using weekly moving average
count_ma <- ts(na.omit(series.weekdays$cnt_ma), frequency=30)
count_ma30 <- ts(na.omit(series.weekdays$cnt_ma30), frequency=30)
decomp <- stl(count_ma30, s.window="periodic")
decomp.week <- stl(count_ma, s.window="periodic")
deseasonal_cnt.week <- seasadj(decomp.week)
deseasonal_cnt <- seasadj(decomp)
plot(decomp, main='Trends') # the remainder cannot be explained through either seasonal or trend components

# breakdown of additional seasonal plots
season.plot <- ggseasonplot(deseasonal_cnt) 
season.plot

#----------------------------------------------#

# statistical testing

# testing whether data display stationarity
adf.test(count_ma30, alternative = "stationary", k=12) # evidence at the 1% level

# auto correlation testing
Acf(count_ma30, main='')
Pacf(count_ma30, main='')

# dickey-fuller test on differenced data
count_d1 <- diff(deseasonal_cnt, differences = 1)
plot(count_d1, ylab = 'Count', xlab='Number of Months', main='Differenced Results')
adf.test(count_d1, alternative = "stationary") # evidence to support alternative hypothesis 

# differenced ACF and PACF
Acf(count_d1, main='ACF for Differenced Series')
Pacf(count_d1, main='PACF for Differenced Series')

# examine coefficients
auto.arima(deseasonal_cnt, seasonal=FALSE)

#----------------------------------------------#

# forecasting

# fit first model
fit.1 <- arima(deseasonal_cnt, order=c(1,1,7))
fcast.1 <- forecast(fit.1, 30) # fit.1 estimates
plot(fcast.1, main='Initial Forecast', ylab='FTE Requirement', xlab='Number of Months') # predictions converge
fcast.1

# fit an ARIMA model of order P, D, Q
fit.2 <- arima(deseasonal_cnt, order=c(0, 1, 1))
fcast.2 <- forecast(fit.2, 30)
plot(fcast.2, main='Forecast with Exponential Smoothing', ylab='FTE Requirement', xlab='Number of Months') # predictions converge at higher value
fcast.2

# allow for drift
fit.3 <- Arima(deseasonal_cnt, order=c(0, 1, 1), include.drift = TRUE)
fcast.3 <- forecast(fit.3, 30)
plot(fcast.3, main='Forcast with Drift', ylab='FTE Requirement', xlab='Number of Months')
fcast.3

# predictive accuracy
accuracy(fit.1)
accuracy(fit.2)
accuracy(fit.3)

# forecast vs actuals allowing for drift
hold <- window(ts(deseasonal_cnt.week), start=230)
fit_no_holdout <- Arima(ts(deseasonal_cnt.week[-c(230:258)]), order=c(0,1,1), include.drift = TRUE)
fcast_no_holdout <- forecast(fit_no_holdout, h=28)
plot(fcast_no_holdout, main="Predictions vs. Actuals", xlab='Number of Days', ylab='FTE Requirement')
lines(ts(deseasonal_cnt.week))

#----------------------------------------------#

# regression analysis

# add days passed column 
series.weekdays$Days <- seq.int(nrow(series.weekdays))
setnames(series.weekdays, old = c("FTE Requirement"), new = c("FTE.Requirement"))

# reg days on number of events
lm.fit <- lm(FTE.Requirement~Days, data = series.weekdays)
summary(lm.fit)

# ggplot function to visualize regression
ggplotRegression <- function (fit) {
  
  ggplot(fit$model, aes_string(x = names(fit$model)[2], y = names(fit$model)[1])) + 
    geom_point() +
    stat_smooth(method = "lm", col = "red") +
    labs(title = paste("Adj R2 = ",signif(summary(fit)$adj.r.squared, 5),
                       "Intercept =",signif(fit$coef[[1]],5 ),
                       " Slope =",signif(fit$coef[[2]], 5),
                       " P =",signif(summary(fit)$coef[2,4], 5))) + ylab("FTE Requirement")+xlab("Days")
}

# plotting regression in ggplot
ggplotRegression(lm.fit)

# still heavily weighted by upper and lower values

# set regression on monthly moving average

# create monthly data set
series.monthly <- series.weekdays[,c("Date","cnt_ma30")]
series.monthly <- na.omit(series.monthly)
series.monthly$Days <- seq.int(nrow(series.monthly))
setnames(series.monthly, old = c("cnt_ma30"), new = c("ARIMA"))

# biuld bi-variate regression
lm.fit.2 <- lm(ARIMA~Days, data = series.monthly)
summary(lm.fit.2)
ggplotRegression(lm.fit.2)

# adding polynomial term
lm.fit.3 <- lm(ARIMA~poly(Days, 4), data = series.monthly)
summary(lm.fit.3)

# plotting the polynomial graph
ggplot(series.monthly, aes(x=Days, y=ARIMA)) + geom_point() + geom_smooth(span=.3) + ggtitle('Polynomial Fit')

#----------------------------------------------#

# using models to predict future FTE Requirement
new.data <- data.frame(Days = c(250, 300, 350))

# predictions
predict(lm.fit.2, newdata = new.data) # prefered predictions 
predict(lm.fit.3, newdata = new.data) # predictions are over fit

#----------------------------------------------#

# advanced modeling

# XGB Forecasting
xgb.monthly <- series.monthly[,c("ARIMA", "Days")]
ARIMA.xgb <- ts(series.monthly$ARIMA)
xgb.fit <- xgbar(ARIMA.xgb)
summary(xgb.fit)
xgb.forecast <- forecast(xgb.fit, h = 100)
plot(xgb.forecast, main = 'XGB Forecast') 

# regularized gradient boosting using train and test splits

# set seed
set.seed(101)

# splitting training set into train and test with a 70/30% split

trainIndex <- createDataPartition(xgb.monthly$ARIMA,
                                  p = .7,
                                  list = FALSE,
                                  times = 1)

# setting train and test sets
xgb.Train <- xgb.monthly[trainIndex,]
xgb.Test <- xgb.monthly[-trainIndex,]

#-----------------------------------------------------------------------#

# training the model

# creating sparse matrix for learning
sparse_matrix_train <- sparse.model.matrix(ARIMA~.-1, data = xgb.Train)

# getting label (outcome), ERP solution dummy vector
xgb.Train$outputVector <- xgb.Train$ARIMA
output_train_vector <- xgb.Train[, "outputVector"]

# building model on training data
bst <- xgboost(data = sparse_matrix_train, label = output_train_vector, max.depth = 10, eta = 1, nthread = 2, nround = 5, 
               objective = "reg:linear")


#-----------------------------------------------------------------------#

# using model on test set to benchmark accuracy

# saving test label
test.Label <- xgb.Test$ARIMA

# transforming test to sparse
sparse_test_matrix <- sparse.model.matrix(ARIMA~.-1, data=xgb.Test)

# getting label (outcome), ERP solution dummy vector from test
xgb.Test$outputVector <- xgb.Test$ARIMA
outputTestVector <- xgb.Test[, "outputVector"]

# making prediction on test data
pred <- predict(bst, sparse_test_matrix)

# set prediction and probabilities as columns 
prediction <- data.frame(pred)

# add columns to test data
xgb.test.final <- cbind(xgb.Test, prediction)

# reorder columns
xgb.test.final <- xgb.test.final[c(2,1,3,4)]
xgb.test.final$outputVector <- NULL

# add columns
xgb.test.final$`Squared diff` <- (xgb.test.final$ARIMA - xgb.test.final$pred)^2
xgb.test.final$`percent error` <- abs((xgb.test.final$pred - xgb.test.final$ARIMA) / xgb.test.final$ARIMA)

# plot actuals vs. predictions
#ggplot(xgb.test.final, aes(x=ARIMA, y=pred)) + geom_point() + ggtitle('Actuals vs. Predictions')
#with(xgb.test.final, plot(Days, ARIMA, type="l", col="red3", 
         #    ylab=("Monthly Moving Average")))

#par(new = T)
#with(xgb.test.final, plot(Days, xgb.test.final$pred, pch=16, axes=F, xlab=NA, ylab=NA, cex=1.2, main = "Actuals vs Predictions"))
#axis(side = 4)
#mtext(side = 4, line = 3, 'Number genes selected')
#legend("topleft",
     #  legend=c("Actuals", "Predictions"),
     #  lty=c(1,0), pch=c(NA, 16), col=c("red3", "black"))

# plotting actuals vs predictions - better method
ggplot(xgb.test.final, aes(Days)) + 
  geom_line(aes(y = ARIMA, colour = "ARIMA")) + 
  geom_point(aes(y = pred, colour = "pred")) + ggtitle("Actuals Vs Predictions") + ylab("FTE Requirement")

#-----------------------------------------------------------------------#

# predicting future FTE Requirement

# days into future
future.requirement <- data.frame(Days = c(250, 300, 350))

# sparse matrix conversion
sparse_matrix_pred <- sparse.model.matrix(~.-1, data=future.requirement)

# making prediction
pred_new_data <- predict(bst, sparse_matrix_pred)
New.predictions <- data.frame(pred_new_data)
New.predictions # predictions again converge

#-----------------------------------------------------------------------#

# statistically significant evidence that FTE Requirement is increasing

# using monthly data
half.1 <- series.monthly[1:114,]
half.2 <- series.monthly[115:228,]

# perform t-test
t.test(half.1$ARIMA, half.2$ARIMA, var.equal=TRUE, paired=FALSE) # evidence that FTE Requirement is in fact increasing

#-----------------------------------------------------------------------#

# isolate predictions
lm.fit.2 <- lm(ARIMA~Days, data = series.monthly)
summary(lm.fit.2)
fit.values <- data.frame(lm.fit.2$fitted.values)

# create new dataframe with actuals and predictions
reg.data <- cbind(series.monthly, fit.values)
reg.data$`percent difference` <- (reg.data$lm.fit.2.fitted.values - reg.data$ARIMA) / reg.data$ARIMA

# create subsets to train best fit regressions
piece.1 <- series.monthly[1:67,]
piece.2 <- series.monthly[68:95,]
piece.3 <- series.monthly[96:162,]
piece.4 <- series.monthly[163:192,]
piece.5 <- series.monthly[193:226,]

# build piecewise regressions
regression.1 <- lm(ARIMA~Days, data = piece.1)
regression.2 <- lm(ARIMA~Days, data = piece.2)
regression.3 <- lm(ARIMA~Days, data = piece.3)
regression.4 <- lm(ARIMA~Days, data = piece.4)
regression.5 <- lm(ARIMA~Days, data = piece.5)

# capture fitted values
piece.fit.1 <- data.frame(regression.1$fitted.values)
piece.fit.2 <- data.frame(regression.2$fitted.values)
piece.fit.3 <- data.frame(regression.3$fitted.values)
piece.fit.4 <- data.frame(regression.4$fitted.values)
piece.fit.5 <- data.frame(regression.5$fitted.values)

# cbind columns
PW.1 <- cbind(piece.1, piece.fit.1)
PW.2 <- cbind(piece.2, piece.fit.2)
PW.3 <- cbind(piece.3, piece.fit.3)
PW.4 <- cbind(piece.4, piece.fit.4)
PW.5 <- cbind(piece.5, piece.fit.5)

# set column names
setnames(PW.1, old = c("regression.1.fitted.values"), new = c("fit"))
setnames(PW.2, old = c("regression.2.fitted.values"), new = c("fit"))
setnames(PW.3, old = c("regression.3.fitted.values"), new = c("fit"))
setnames(PW.4, old = c("regression.4.fitted.values"), new = c("fit"))
setnames(PW.5, old = c("regression.5.fitted.values"), new = c("fit"))

# crease final step function
piece.wise <- rbind(PW.1, PW.2, PW.3, PW.4, PW.5)

# plotting actuals vs predictions
ggplot(piece.wise, aes(Days)) + 
  geom_point(aes(y = ARIMA, colour = "ARIMA")) + 
  geom_line(aes(y = fit, colour = "fit")) + ggtitle("Step Function") + ylab("FTE Requirement")


# end of script