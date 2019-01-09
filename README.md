# time-series

t=ts(a$Allocation,start = c(2018,1),frequency = 12)
r=stl(t,s.window = "periodic",robust=TRUE)# for additive
bx=BoxCox(t,lambda=0)# for multiplicative
r=stl(t,s.window = "periodic",robust=TRUE)# for multiplicative
tsplot(t,margins = 1)
seasonplot(t)
monthplot(t)
lag.plot(t)
lag.plot(t,9)
ggAcf(t)# it show autocorrelation for trend in decreasing format while for seasonal,it show on seasonal period.if it is not significant then it show series is white noise.For a stationary time series, the ACF will drop to zero relatively quickly, while the ACF of non-stationary data decreases slowly.for non-stationary data, the value of  
r1 is often large and positive. 
beerfit1 <- meanf(t, h=11)
beerfit2 <- naive(t, h=11)
beerfit3 <- snaive(t, h=11)
beerfit3 <- rwf(t, h=11,drift=TRUE)
ma2x4 <- ma(t, order=4, centre=TRUE)# for even


# forecast by seasonal decompostion
r=stl(t,s.window = "periodic")
eeadj <- seasadj(r)
fit=naive(eeadj)
plot(naive(eeadj), xlab="New orders index",
  main="Naive forecasts of seasonally adjusted data")
fcast <- forecast(fit, method="naive")
s=snaive(r$time.series[,"seasonal"])
forecast=s$mean+fcast$mean
#Simple exponential method
s=ses(t,h=2)
# holt method with trend
h=holt(t,h=5,alpha=0.8,beta=0.3)
# holt method with damped
h=holt(t,h=5,alpha=0.8,beta=0.3)
# ARIMA modelling
d=diff(log(t),lag=12,1)# for seasonal data
d=diff(log(t)) # for non seasonal data
ac=ggacf(t)
pac=ggpacf(t)
tsdisplay(diff(eeadj),main="")
Box.test(residuals(fit), lag=24, fitdf=4, type="Ljung") # portmanteau test


r=seas(t,x11="")
trecy=trendcycle(r,series="Trend")
seaad=seasadj(r,series="seasonally-adjusted data")
r %>% seasonal() %>% ggsubseriesplot() + ylab("Seasonal")
