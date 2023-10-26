# Implementing GP Regression
# Algorithm 2.1 

# Squared Exponential kernel or otherwise called 
# Covariance Function

SquaredExpKernel <- function(x1,x2,sigmaF= 1,l=0.3){
  n1 <- length(x1)
  n2 <- length(x2)
  K <- matrix(NA,n1,n2)
  for (i in 1:n2){
    K[,i] <- sigmaF^2*exp(-0.5*( (x1-x2[i])/l)^2 )
  }
  return(K)
}

posteriorGP <- function(X, y, XStar, sigmaNoise, k, ...){
  n = length(X)
  K = k(X,X, ...)
  Kstar =k(X,XStar, ...)
  L = t(chol(K+((sigmaNoise**2)*diag(n))))

  alph <- solve(t(L), solve(L,y))

  # Predictive Mean
  fStar <- t(Kstar) %*% alph

  # Predictive Variance
  v <- solve(L, Kstar)
  var_fStar <- k(XStar, XStar)- t(v) %*% v
  # log_marg_ll <- (-0.5*(t(y) %*% alph)) -(det(L)*det(t(L))) - (0.5*n*log(2*pi))
  return(list('fStar' = fStar, 'var_fStar' = var_fStar))
}


# sub question 2
first_data <- c(0.4 ,0.719)
xstar = seq(-1,1,length=100)
sigmaNoise=0.1
k = SquaredExpKernel
post_GP <- posteriorGP(first_data[1], first_data[2], xstar, sigmaNoise, k)

mean_PostPred <- post_GP$fStar
var_PostPred <- post_GP$var_fStar

plot(xstar,mean_PostPred,col='red',ylab='pos mean',type='l', ylim = c(-3,5))
lines(xstar, mean_PostPred + 1.96*sqrt(diag(var_PostPred)), lty = 5)
lines(xstar, mean_PostPred - 1.96*sqrt(diag(var_PostPred)), lty = 5)
points(first_data[1], first_data[2], pch = 23, col = 'red',  bg = 'blue')
legend("topright", legend = c("Predictive Mean", "Predictive Variance", "Observed Data"),
       col = c("red", "black",'blue'), lty = c(1, 2, NA), pch = c(NA, NA, 16))
# sub question 3

x_2 <- c(0.4, -0.6)
y_2 <- c(0.719, -0.044)
post_GP2 <- posteriorGP(x_2, y_2, xstar, sigmaNoise, k)

mean_PostPred2 <- post_GP2$fStar
var_PostPred2 <- post_GP2$var_fStar

plot(xstar,mean_PostPred2,col='red',ylab='pos mean',type='l', ylim = c(-3,5))
lines(xstar, mean_PostPred2 + 1.96*sqrt(diag(var_PostPred2)), lty = 5)
lines(xstar, mean_PostPred2 - 1.96*sqrt(diag(var_PostPred2)), lty = 5)
points(x_2, y_2, pch = 23, col = 'red', bg = 'blue')
legend("topright", legend = c("Predictive Mean", "Predictive Variance", "Observed Data"),
       col = c("red", "black",'blue'), lty = c(1, 2, NA), pch = c(NA, NA, 16))
# sub question 4

x_3 <- c(-1, -0.6, -0.2, 0.4, 0.8)
y_3 <- c(0.768, -0.044, -0.940, 0.719, -0.664)

post_GP3 <- posteriorGP(x_3, y_3, xstar, sigmaNoise, k)

mean_PostPred3 <- post_GP3$fStar
var_PostPred3 <- post_GP3$var_fStar

plot(xstar,mean_PostPred3,col='red',ylab='pos mean',type='l', ylim = c(-3,3))
lines(xstar, mean_PostPred3 + 1.96*sqrt(diag(var_PostPred3)), lty = 5)
lines(xstar, mean_PostPred3 - 1.96*sqrt(diag(var_PostPred3)), lty = 5)
points(x_3, y_3, pch = 23, col = 'red', bg = 'blue')
legend("topright", legend = c("Predictive Mean", "Predictive Variance", "Observed Data"),
       col = c("red", "black",'blue'), lty = c(1, 2, NA), pch = c(NA, NA, 16))
# sub question 5

x_4 <- c(-1, -0.6, -0.2, 0.4, 0.8)
y_4 <- c(0.768, -0.044, -0.940, 0.719, -0.664)

post_GP4 <- posteriorGP(x_4, y_4, xstar, sigmaNoise, k, sigmaF = 1, l = 1)

mean_PostPred4 <- post_GP4$fStar
var_PostPred4 <- post_GP4$var_fStar

plot(xstar,mean_PostPred4,col='red',ylab='pos mean',type='l', ylim = c(-3,3))
lines(xstar, mean_PostPred4 + 1.96*sqrt(diag(var_PostPred4)), lty = 5)
lines(xstar, mean_PostPred4 - 1.96*sqrt(diag(var_PostPred4)), lty = 5)
points(x_4, y_4, pch = 23, col = 'red', bg = 'blue')
legend("topright", legend = c("Predictive Mean", "Predictive Variance", "Observed Data"),
       col = c("red", "black",'blue'), lty = c(1, 2, NA), pch = c(NA, NA, 16))

# Question 2
# sub question 1
# GP Regression with KernLab
library(kernlab)
data <- read.csv("https://github.com/STIMALiU/AdvMLCourse/raw/master/GaussianProcess/Code/TempTullinge.csv", header=TRUE, sep=";")

#Consider only time = 1, 6, 11, . . ., 2186
time <- 1:nrow(data) #1... 2190
day <- day <- (1:nrow(data))%%365 #1...365 1...365
day[day == 0] <- 365
indx <-  seq(1, length(time), 5)

time_sub <- time[indx]
day_sub <- day[indx]
temp_sub <- data[indx,2]

# train_data <- data
# train_data$time <- c(1:nrow(data))
# train_data$time2 <- train_data$time**2
# train_data$day <- rep(c(1:365), 6)

SEkernel <- function(l = 1, sigmaF = 1) {
  kernval <- function(x, x_prime = NULL) {
    res <- matrix(0, nrow = length(x), ncol = length(x_prime))
    for (i in 1:length(x)) {
      for (j in 1:length(x_prime)) {
        res[i, j] <- (x[i] - x_prime[j])^2
      }
    }
    res <- (sigmaF^2) * exp(-res / (2 * l^2))
    return(res)
  }
  class(kernval) <- 'kernel'
  return(kernval) 
}


SEkernel_func <- SEkernel(1, 1)
# Evaluating the kernel at x=1 and x' = 2
cat("Kernel evaluation at x = 1 and x' = 2:",SEkernel_func(x = 1, x_prime = 2))

# Computing covariance matrix K(X, Xstar) 
kernelMatrix(kernel = SEkernel_func, x = c(1,3,4), y = c(2, 3, 4))

# Sub Question 2

lm_fit <- lm(temp_sub~time_sub+I(time_sub**2))
sigma2_resid <- var(lm_fit$residuals)
se_kern <- SEkernel(l = 0.2, sigmaF = 20)

GPfit <- gausspr(time_sub, temp_sub, 
                 kernel = se_kern,
                 variance.model = TRUE,
                 var = sigma2_resid)

meanPred  <- predict(GPfit, time_sub)
sdPred <- predict(GPfit,time_sub, type="sdeviation")
plot(time_sub, temp_sub, type = 'l',ylim = c(-60, 60))
lines(time_sub, meanPred, col = 'red', lwd=2.0)
# We use 0.1 because of a bug in the kernlab
# Using 1.96 makes the CI bands to be too wide
lines(time_sub, meanPred+1.96*sdPred,col="blue", lwd = 2.0)
lines(time_sub, meanPred-1.96*sdPred,col="blue", lwd = 2.0)
legend("topright", legend = c("Predictive Mean - fStar", "95% CI for fStar"),
       col = c("red",'blue'), lty = c(1, 1))

# Sub Question 3
train_data_scaled <- train_data
train_data_scaled$temp <- scale(train_data_scaled$temp)[,1]
train_data_scaled$time <- scale(train_data_scaled$time)[,1]
train_data_scaled$time2 <- scale(train_data_scaled$time2)[,1]

#Scale the x and y -> (val-mean)/sd
scaled_time_sub <- scale(time_sub)
scaled_time <- scale(time)
scaled_temp_sub <- scale(temp_sub)

lm_scaled <- lm(temp~time+time2, data = train_data_scaled)

sigma2_scale <- var(lm_scaled$residuals)
post_GP_temp <- posteriorGP(X = train_data_scaled$time, 
                            y = train_data_scaled$temp, 
                            XStar = train_data_scaled$time, 
                            sigmaNoise = sqrt(sigma2_scale), k = se_kern)

unscaled_post_mean_temp <- (post_GP_temp$fStar*sd(train_data$temp))+mean(temp_sub)
unscaled_post_var_temp<- sqrt(diag(post_GP_temp$var_fStar))*sd(temp_sub)


plot(time_sub, temp_sub, type = 'l', ylim = c(-30, 60), xlab = 'Time', ylab = 'Temperature')
lines(time_sub, unscaled_post_mean_temp, col = 'red', lwd = 2.0)
lines(time_sub,  unscaled_post_mean_temp+ 1.96*unscaled_post_var_temp, col = 'blue', lwd = 2.0)
lines(time_sub,  unscaled_post_mean_temp- 1.96*unscaled_post_var_temp, col = 'blue', lwd = 2.0)
#Predictive interval for yStar
lines(time_sub,  unscaled_post_mean_temp+ 1.96*(unscaled_post_var_temp + sqrt(sigma2_scale)), col = 'green', lwd = 2.0, lty = 2)
lines(time_sub,  unscaled_post_mean_temp- 1.96*(unscaled_post_var_temp + sqrt(sigma2_scale)), col = 'green', lwd = 2.0, lty = 2)
legend("topright", legend = c("Predictive Mean - fStar", "95% CI fStar", "95% CI yStar"),
       col = c("red", "blue", "green"), lty = c(1, 1, 2))

# How to sample ystar?
# plot(train_data$time, train_data$temp, type = 'l', ylim = c(-25, 35))
# lines(train_data$time, mean_postPred_temp + rnorm(n = length(mean_postPred_temp),mean = 0, sqrt(sigma2_resid)), col = 'gray', lty = 2)

# sub question 4

lm_fit2 <- lm(temp~day+(day**2), data = train_data)
sigma2_resid2 <- var(lm_fit2$residuals)

GPfit_2 <- gausspr(train_data$day, train_data$temp, 
                 kernel = se_kern,
                 variance.model = TRUE,
                 var = sigma2_resid)

meanPred2  <- predict(GPfit_2, train_data$day)
sdPred2 <- predict(GPfit_2,train_data$day, type="sdeviation")

plot(train_data$time, data$temp, type = 'l', ylim = c(-25, 40))
lines(train_data$time, meanPred2, col = 'red', lwd=2.0)
# We use 0.1 because of a bug in the kernlab
# Using 1.96 makes the CI bands to be too wide
lines(train_data$time, meanPred2+0.1*sdPred2,col="blue", lwd = 2.0)
lines(train_data$time, meanPred2-0.1*sdPred2,col="blue", lwd = 2.0)
lines(train_data$time, meanPred, col = 'green', lwd = 2.0)
legend("topright", legend = c("Predictive Mean - fStar - Day", 
                              "95% CI f_star", 
                              "Predictive Mean - fStar - Time"),
       col = c("red", "blue", 'green'), 
       lty = c(1, 1, 1))

# sub question 5

SE_periodic_kernel <- function(l1 = 1, l2 = 1, sigmaF = 1, d = 1){
  kern_val <- function(x, x_prime){
    res <- matrix(0, nrow = length(x), ncol = length(x_prime))
    for (i in 1:length(x)) {
      for (j in 1:length(x_prime)) {
        res[i, j] <- abs(x[i] - x_prime[j])
      }
    }
    return((sigmaF**2)*exp(-(2*sin((pi*res)/d)**2)/(l1**2))*exp(-(res**2)/(2*l2**2)))
  }
  class(kern_val) <- 'kernel'
  return(kern_val)
}
se_per_kern <- SE_periodic_kernel(l1 = 1, l2 = 10, sigmaF = 20, d = 365/sd(train_data$time))

GP_fit3 <- gausspr(train_data$time, train_data$temp, 
                   kernel = se_per_kern,
                   variance.model = TRUE,
                   var = sigma2_resid)

meanPred3  <- predict(GP_fit3, train_data$time)
sdPred3 <- predict(GP_fit3,train_data$time, type="sdeviation")

plot(train_data$time, train_data$temp, type = 'l', ylim = c(-25, 50))
lines(train_data$time, meanPred3, col = 'red', lwd=2.0)
# We use 0.1 because of a bug in the kernlab
# Using 1.96 makes the CI bands to be too wide
lines(train_data$time, meanPred3+0.1*sdPred3,col="blue", lwd = 1.5)
lines(train_data$time, meanPred3-0.1*sdPred3,col="blue", lwd = 1.5)
# lines(train_data$time, meanPred2, col = 'green', lwd = 1.5, lty = 2)
legend("topright", legend = c("Predictive Mean - fStar", "95% CI for fStar"),
       col = c("red",'blue'), lty = c(1, 1))

# Question 3
# GP Classification
library(AtmRay)
# sub question 1
data2 <- read.csv("https://github.com/STIMALiU/AdvMLCourse/raw/master/GaussianProcess/Code/banknoteFraud.csv", header=FALSE, sep=",")
names(data2) <- c("varWave","skewWave","kurtWave","entropyWave","fraud")
data2[,5] <- as.factor(data2[,5])

set.seed(111)
SelectTraining <- sample(1:dim(data2)[1], size = 1000,
                                        replace = FALSE)

train_data1 <- data2[SelectTraining, ]
test_data1 <- data2[-SelectTraining, ]

train_data1 <- train_data1[,c(1,2,5)]
test_data1 <- test_data1[,c(1,2,5)]

GPfitFraud <- gausspr(fraud~varWave+skewWave, data = train_data1)

train_pred <- predict(object = GPfitFraud, train_data1[,1:2])
conf_mat_train <- table(train_data1$fraud, train_pred)
train_acc <- sum(diag(conf_mat_train))/sum(conf_mat_train)

probPreds <- predict(GPfitFraud, train_data1[,1:2], type = 'probabilities')

x1 <- seq(min(train_data1[,1]),max(train_data1[,1]),length=100)
x2 <- seq(min(train_data1[,2]),max(train_data1[,2]),length=100)
gridPoints <- meshgrid(x1, x2)
gridPoints <- cbind(c(gridPoints$x), c(gridPoints$y))
gridPoints <- data.frame(gridPoints)
names(gridPoints) <- names(train_data1)[1:2]

probPreds <- predict(GPfitFraud, gridPoints, type="probabilities")

# Plotting for Prob(fraud = 1)
contour(x1,x2,matrix(probPreds[,2],100,byrow = TRUE), 20, xlab = "varWave", 
        ylab = "skewWave", main = 'Prob(fraud)')
points(train_data1[train_data1[,3]==1,1],train_data1[train_data1[,3]==1,2],col="blue", pch = 16)
points(train_data1[train_data1[,3]==0,1],train_data1[train_data1[,3]==0,2],col="red", pch = 16)

# sub question 2
test_pred <- predict(object = GPfitFraud, test_data1[,1:2])
conf_mat_test <- table(test_data1$fraud, test_pred)
test_acc <- sum(diag(conf_mat_test))/sum(conf_mat_test)

# sub question 3

train_data2 <- data2[SelectTraining, ]
test_data2 <- data2[-SelectTraining,]

GPfitFraud2 <- gausspr(fraud~., data = train_data2)
train_pred2 <- predict(object = GPfitFraud2, train_data2[,1:4])

conf_mat_train2 <- table(train_data2$fraud, train_pred2)
train_acc2 <- sum(diag(conf_mat_train2))/sum(conf_mat_train2)

test_pred2 <- predict(object = GPfitFraud2, test_data2[,1:4])
conf_mat_test2 <- table(test_data2$fraud, test_pred2)
test_acc2 <- sum(diag(conf_mat_test2))/sum(conf_mat_test2)
