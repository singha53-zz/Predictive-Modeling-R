Analyzing Trends in Employee Requirement - (FTE in the R Code which stands for Full Time Employee)
--------------------------------------------------------------------------------------------------

The purpose of this report is to analyze trends in Employee Requirement. Essentially, we have time entry for a firm from each of its employees on a daily basis. We want to determine how this time entry allows us to forecast the number of employees the firm needs to adequately complete its business processes without overworking its employees. Building accurate forecasts allows firms a better sense of their business operations and the ability to accurately plan for the future. We use a number of methods for time series analysis in R to do this. The packages tseries, forecast, and AnomalyDetection are all helpful with time series analysis and are used extensively in this demonstration.

Plotting Employee Requirement using Weekdays only
-------------------------------------------------

Employees entered very little time on the weekends, as we would expect. 2016-04-02 and 2016-04-03 were the first Saturday and Sunday in April 2016. As you will see later, the inclusion of weekends into the graph create consistent troughs.

    ##         Date   Newtime Time.Extrapolated FTE Requirement
    ## 1 2016-04-01 112731.00       41146815.00     328.4388170
    ## 2 2016-04-02    114.00          41610.00       0.3321360
    ## 3 2016-04-03    225.75          82398.75       0.6577167
    ## 4 2016-04-04 236684.25       86389751.25     689.5733657
    ## 5 2016-04-05 284052.00      103678980.00     827.5780650
    ## 6 2016-04-06 265731.00       96991815.00     774.2003115

In the graph below, we exclude weekends from the data, along with significant outliers, which we observe around Holidays. We include a trend line in the graph below.

![](ARIMA_Forecasting_files/figure-markdown_github/viz1-1.png)

We still observe troughs in the graph in association with Holidays. Nevertheless, the general trend shows an increase in Employee Requirement over the last year given the amount of work the employees of the firm were completing.

Applying Anomaly Detection to FTE Requirement per Weekday
---------------------------------------------------------

We next apply anomaly detection to determine whether any of the significant troughs visible in the data represent anomalies.

![](ARIMA_Forecasting_files/figure-markdown_github/viz2-1.png)

Anomaly detection considered only two points to be anomalies, meaning that less than 1% of total observations are classified as anomalous after excluding weekends.

Analyzing Trends while keeping all observations
-----------------------------------------------

While excluding weekends from the data helps eliminate many of the toughs we observe, we still have the question of what to do with the time entered on the weekends. Though little in comparison to time entered on weekdays, it still represents thousands of minutes which should be included, if possible, in the analysis. Below we graph the entire set of 365 days in the data, including weekends.

![](ARIMA_Forecasting_files/figure-markdown_github/viz3-1.png)

As we see, time entry follows a relatively predictable series of peaks during mid week, and troughs on the weekends.

Auto Regressive Integrated Moving Average (ARIMA)
-------------------------------------------------

Neither keeping nor removing weekends is perfectly ideal for time series analysis, as excluding weekends leads to a loss of time entered and keeping them leads to high fluctuation which proves difficult to model. A way around modeling or excluding the troughs posed by the weekend Employee Requirement values is to use a moving weekly or monthly average with respect to Employee REquirement. Moving averages are calculated using auto regressive integration (which is less complex than it might sound). Below you will find graphs of the weekly and monthly moving averages on top of the total sets of observations, and then graphed alone.

Ultimately, the predictions we generated when including weekends in the analysis negatively impacted the Employee Requirement predictions. The predictions returned were far lower than we could reasonably expect, given they were being pulled downwards by the low weekend Employee Requirement values. This being the case, we exclude weekends when calculating the weekly and monthly moving averages.

    ## [1] 747.5118

![](ARIMA_Forecasting_files/figure-markdown_github/viz4-1.png)![](ARIMA_Forecasting_files/figure-markdown_github/viz4-2.png)

Trend Breakdown
---------------

Using the weekly moving average, we can observe trends which appear in the Employee Requirement over time. We do just this in the plot below, which looks at overall trends in the series, seasonal fluctuations, and variation which is unexplained by either the seasonal or general trends.

![](ARIMA_Forecasting_files/figure-markdown_github/viz5-1.png)

As we see from the trend portion of the seasonal breakdown, the analysis shows a significant increase in average Employee Requirement using monthly average from around 725 to around 900. The increase in Employee Requirement is not uniformly distributed, which may suggest a custom step function is appropriate for forecasting.

Statistical Testing for Stationarity and Auto-Correlation
---------------------------------------------------------

We next test to determine whether the series demonstrates stationarity, which is to say, the series retains mean, variance, and auto-correlation over time.

    ## 
    ##  Augmented Dickey-Fuller Test
    ## 
    ## data:  count_ma30
    ## Dickey-Fuller = -2.6005, Lag order = 12, p-value = 0.3238
    ## alternative hypothesis: stationary

![](ARIMA_Forecasting_files/figure-markdown_github/viz6-1.png)![](ARIMA_Forecasting_files/figure-markdown_github/viz6-2.png)

The statistical results align with what we observe in the data. We cannot reject non-stationarity, which is reasonable given the generally increasing trend we observe in Employee Requirement. We would not expect mean, variance, and auto-correlation to remain unchanged over time. We also observe sinasoidal patterns in auto-correlation and partial auto-correlation, which again match with the observed patterns in the monthly moving average for Employee Requirement.

Forecasting
-----------

We now examine different forecasting techniques for predicting future Employee Requirement using moving monthly averages. The first forecast below uses exponential smoothing.

![](ARIMA_Forecasting_files/figure-markdown_github/viz7-1.png)

The initial forecast converges on a monthly moving average of 858 Employee Requirement. This seems to be a suboptimal prediction given that the trend in Employee Requirement is generally increasing. The smoothing technique negated the upward trend. We can correct this by allowing for 'drift.'

![](ARIMA_Forecasting_files/figure-markdown_github/viz8-1.png)

As seen, including drift allows the forecast to display an increasing trend. In this case, the forecast looks to be increasing at a linear rate. We next examine the drift forecast's performance when measured against the observed FTE Requirement over the final 21 days present in the data while using weekly moving average.

![](ARIMA_Forecasting_files/figure-markdown_github/viz9-1.png)

The forecast performs well over the final 21 days with repsect to predicting Employee Requirement. Using the predictions of the drift model is one option for predicting future Employee Requirement. Another would be to build custom step functions. Under either scenario, we can move from orders due to new customers to Employee requirement. For a simple example, see below:

orders: o(c) = 5c, where c = additional customers

Employee Requirement: e(o) = .001(o) + 600, where o = additional orders

To move from new customers to Employee Requirement we can model:

e(o(c)) = .001(5c) + 600, or -&gt; e(o(c)) = .005c + 600

For each additional customer, Employee requirement will increase by .005.

Regression Analysis
-------------------

We next use regression analyis to model predictions for future Employee Requirement. We will use linear, polynomial, and non-parametric models. Each of the models basically consider Employee Requirement to be a function of time. The weekly Employee Requirement trend data is used in the graph below, with the model results listed for review above the graph.

    ## 
    ## Call:
    ## lm(formula = FTE.Requirement ~ Days, data = series.weekdays)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ## -807.5 -252.9  114.0  195.6  323.9 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) 658.1303    32.2335  20.418  < 2e-16 ***
    ## Days          0.6913     0.2174   3.179  0.00166 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 257.1 on 254 degrees of freedom
    ## Multiple R-squared:  0.03827,    Adjusted R-squared:  0.03448 
    ## F-statistic: 10.11 on 1 and 254 DF,  p-value: 0.00166

![](ARIMA_Forecasting_files/figure-markdown_github/viz10-1.png)

The model equation is: Employee Requirement = 658 + .69(Days). This gives the predicted Employee Requirement N days (excluding weekends) after April 1. For example, the predicted FTE Requirement 100 weekdays after April 1, 2016 is calculated as: 658 + .69(100) = 728. A second option is to use the monthly moving average. The monthly moving average reveals a sinasoid like trend not visible in the weekly moving average. The regression results and graph are displayed below:

    ## 
    ## Call:
    ## lm(formula = ARIMA ~ Days, data = series.monthly)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -109.768  -33.389    7.329   33.555   80.769 
    ## 
    ## Coefficients:
    ##              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) 666.81571    6.34100   105.2   <2e-16 ***
    ## Days          0.66379    0.04844    13.7   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 47.51 on 224 degrees of freedom
    ## Multiple R-squared:  0.4561, Adjusted R-squared:  0.4536 
    ## F-statistic: 187.8 on 1 and 224 DF,  p-value: < 2.2e-16

![](ARIMA_Forecasting_files/figure-markdown_github/viz11-1.png)

The model equation is: 667 + .66(Days). So, 100 weekdays after April 1, 2016, the predicted FTE Requirement is: 667 + .66(100) = 733. The two predictions are only 5 FTE apart. We can solve the system of equations to determine when they would be equal, or in other words, match their predictions with respect to FTE Requirement:

667 + .66D = 658 + .69D: 9 = .03D 300 = D, so the two predictions would be equal 300 weekdays after April 1, 2016. Predictions going out longer than this would be unlikely without remodeling the relationship to account for additional data.

We can also include polynomial terms to determine whether polynomial regression helps better predict than simple linear regression.

    ## 
    ## Call:
    ## lm(formula = ARIMA ~ poly(Days, 4), data = series.monthly)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ## -87.76 -36.70  10.90  26.13  81.58 
    ## 
    ## Coefficients:
    ##                Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)     742.156      2.564 289.410  < 2e-16 ***
    ## poly(Days, 4)1  651.028     38.551  16.887  < 2e-16 ***
    ## poly(Days, 4)2  370.724     38.551   9.616  < 2e-16 ***
    ## poly(Days, 4)3   40.609     38.551   1.053    0.293    
    ## poly(Days, 4)4  194.873     38.551   5.055 9.02e-07 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 38.55 on 221 degrees of freedom
    ## Multiple R-squared:  0.6466, Adjusted R-squared:  0.6402 
    ## F-statistic: 101.1 on 4 and 221 DF,  p-value: < 2.2e-16

    ## `geom_smooth()` using method = 'loess'

![](ARIMA_Forecasting_files/figure-markdown_github/viz12-1.png)

The high order polynomial term does in fact model the data quite closely, and of course reduces the error in comparison to the linear regression on the training data. However, the polynomial terms quickly yeild far higher predictions than can reasonably be expected. This becomes more pronounced the farther into the future we model. Below we compare estimates for 250, 300, and 350 days beyond betewen the linear and polynomial models:

    ##        1        2        3 
    ## 832.7637 865.9533 899.1429

    ##        1        2        3 
    ## 1108.545 2006.379 4016.335

The linear predictions are seen on top with the polynomial predictions below. As you see, the polynomial model's predictions begin to increase exponentially and are therefore of limited use as we predict out further and further.

Non-Parametric Methods
----------------------

We will lastly consider two versions of regularized gradient boosting for predicting Employee Requirement. Regularized gradient boosting is known for its exceptional acuracy. However, the algorithm typically performs best with multi-dimensional data. In our context, using only time and FTE Requirement, it is unclear whether 'xgboost' will pick up on trends. See the first iteration of the xgboost model below:

![](ARIMA_Forecasting_files/figure-markdown_github/viz13-1.png)

The predictions are seen in blue, and extend 30 weekdays past the ending date of the observed data. The model is unable to pick up on the generally increasing trend in FTE Requirement. This is noted in the graph below which projects out the xgboost model 100 weekedays using the monthly moving average.

![](ARIMA_Forecasting_files/figure-markdown_github/viz14-1.png)

The xgboost model is unable to accurately account for the general pattern of increase in FTE Requirement. The next option is to train and test an xgboost model in the regression context, and then use the trained model to predict future points.

    ## [1]  train-rmse:63.690445 
    ## [2]  train-rmse:9.181771 
    ## [3]  train-rmse:4.087582 
    ## [4]  train-rmse:3.035386 
    ## [5]  train-rmse:1.758917

![](ARIMA_Forecasting_files/figure-markdown_github/viz15-1.png)

Being a non-parametric algorithm, the xgboost model returns exceptionally accurate Employee Requirement predictions for the training data. We next test the xgboost model on unobserved future data points, 250, 300, and 350 weekdays from April 1, 2016.

    ##   pred_new_data
    ## 1      860.8329
    ## 2      860.8329
    ## 3      860.8329

Unfortunately, the xgboost model is unable to pick up on the general increasing trend in Emploee Requirement and returns predictions that converge quickly.

Is the increase in FTE Requirement Statistically Significant?
-------------------------------------------------------------

We observe a general pattern of increase in Employee Requirement with respect to monthly moving average. We now test whether the increasing pattern is statistically significant. We are looking to determine whether there is statistically significant evidence that the average Employee Requirement has increase. To do this, we split the monthly data into a first and second half, and perform a t-test.

    ## 
    ##  Two Sample t-test
    ## 
    ## data:  half.1$ARIMA and half.2$ARIMA
    ## t = -12.352, df = 223, p-value < 2.2e-16
    ## alternative hypothesis: true difference in means is not equal to 0
    ## 95 percent confidence interval:
    ##  -94.95408 -68.82395
    ## sample estimates:
    ## mean of x mean of y 
    ##  701.2673  783.1563

There is evidence at the 1% level of significance that the average FTE Requirement over the first 113 weekdays of the data is not equal to the average FTE Requiremnt over the second 113 weekdays. This is strong evidence that FTE Requirement is increasing over time.

Step Functions for Prediction
-----------------------------

The final method we will cover for predicting future Employee Requirement is to create a custom Step Function.See the graph of the custom step function below:

![](ARIMA_Forecasting_files/figure-markdown_github/viz16-1.png)

We see that between days 60 and 95 FTE Requirement drops off significnatly. This is during the summer months, and hits a low around July 4th. Similarly, the FTE Requirement drops off significantly around the winter holidays, and increases rapidly from there. Using step functions may allow us to more accurately model future Employee Requirement by taking into account seasonal fluctuations that are always pciked up by forecasting algorithms.

Summary of Results
------------------

This report has highlighted forecasting methods for predcting future Employee Requirement using time series analysis. The most promising results are produced using 'drift' time series, linear models, and step functions. Although non-parametric models are highly efficient in minimizing training error, the low dimensionality of the data, and relatively few data points, hindered their predictive accuracy when looking forward.

The best course of action for predicting baseline increase in Employee Requirement is to use linear, drift, or step function models. A combination of the three can also be used. When composed with the function for Employee Requirement increase given additional customers, an accurate forecasted result can be expected, allowing the firm to plan for and justify staffing increases with high accuracy.
