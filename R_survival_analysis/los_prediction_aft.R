# Script for LOS prediction using Accelerated Failure Time Model

# Set working directory
setwd("~/Johns Hopkins University/2019-20/Fall 2019/Precision Care Medicine I/TBI_Project/tbi/R_survival_analysis")

library(tidyverse)
library(knitr)
library(ciTools)
library(here)
library(survival)
library(standardize)
library(BART)
library(MASS)
library(mlbench)
library(caret)
require(MASS)
library(TBSSurvival)

set.seed(20180925)

#------------------- LOS ----------------------#
mort_status <- 'alive'

# Loading data
los = read.table("../notebooks/los_surv_analysis_dat.csv", sep=",", header=TRUE)
los$GCS <- NULL
los$Value <- NULL
los$patientunitstayid <- NULL
los$X <- NULL

los <- los[los['los'] <= 30,]

# # Keeping either only dead or alive patients
# if (mort_status == 'alive') {
#   los <- los[los['death'] == 'False',]
# } else if (mort_status == 'dead') {
#   los <- los[los['death'] == 'True',]
# }

los$death <- NULL

# Splitting data into train and test
smp_size <- floor(0.8 * nrow(los))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(los)), size = smp_size)

train <- (los[train_ind, ])
test <- (los[-train_ind, ])
val_size <- floor(0.25 * nrow(train))
val_ind <- sample(seq_len(nrow(train)), size = val_size)
val <- train[val_ind,]
train <- train[-val_ind,]

los_train <- train$los
occurs_train <- train$occurs
los_val <- val$los
occurs_val <- val$occurs
los_test <- test$los
occurs_test <- test$occurs


normalize <- function(x) { 
  z=x
  if(min(x)<max(x)){ 
    z=(x - min(x)) / (max(x) - min(x))
  }
  return(z)
}  

train_norm <- as.data.frame(apply(train, 2, normalize))
val_norm <- as.data.frame(apply(val, 2, normalize))
test_norm <- as.data.frame(apply(test, 2, normalize))

train_norm$los <- los_train
train_norm$occurs <- occurs_train
val_norm$los <- los_val
val_norm$occurs <- occurs_val
test_norm$los <- los_test
test_norm$occurs <- occurs_test

train <- train_norm
val <- val_norm
test <- test_norm

# # ------- Resampling the Train dataset ---------- #
# 
# # Setting resampling method
# # 'mean' or 'exp_decay'
# resample_method = 'exp_decay'
# 
# # To keep the same number of people in each bin
# histogram(train$los)
# intervals <- seq(from=3, to=30, by=3)
# temp0 <- train[(train['los'] >= 0) & (train['los'] <= 3),]
# cts <- c(nrow(temp0))
# for (i in intervals[2:length(intervals)]) {
#   temp <- train[(train['los'] > i-3) & (train['los'] <= i),]
#   print(nrow(temp))
#   cts <- c(cts, nrow(temp))
# }
# 
# # ------ Trying exponential decay for resampling ---- #
# fun1 <- function(a,r,t) { 
#   return(a*exp(-r*t))
# }
# # ----------------------------------------------------#
# 
# # Getting number of elements per bin
# if (resample_method == 'mean') {
#   binsize <- rep(ceiling(mean(cts)), length(intervals))
# } else if (resample_method == 'exp_decay') {
#   t <- seq(from=0, to=30-3, by=3)
#   binsize <- fun1(260,0.05,t)  
# } else {
#   stop(paste("Invalid resampling method:", resample_method))
# }
# 
# temp0 <- train[(train['los'] >= 0) & (train['los'] <= 3),]
# rand_idcs <- round(runif(binsize[1])*nrow(temp0))
# train_resamp <- temp0[rand_idcs,]
# 
# j = 2
# for (i in intervals[2:length(intervals)]) {
#   temp <- train[(train['los'] > i-3) & (train['los'] <= i),]
#   rand_idcs <- round(runif(binsize[j])*nrow(temp))
#   train_resamp <- rbind(train_resamp, temp[rand_idcs,])
#   j = j+1
# }
# 
# train <- train_resamp
# 
# los_train <- train$los
# occurs_train <- train$occurs
# los_val <- val$los
# occurs_val <- val$occurs
# los_test <- test$los
# occurs_test <- test$occurs
# 
# # -------- End of resampling --------- #

dist="loglogistic"

# Using variable selection

# Fitting model
(fit <- survreg(Surv(los,occurs) ~ . -los -occurs, data = train, dist=dist))
# n <- names(train)
# f <- as.formula(paste("Surv(train$los,train$occurs) ~", paste("train$",n[!n %in% "train"], collapse = " + "), "-train$los -train$occurs"))
# (fit <- tbs.survreg.be(f))

# Getting p-values for each feature and choosing top n
n=9 # 12 for no resampling, 11 with exponential resampling (but singularities occur)
tb <- data.frame(summary(fit)$table)
tb <- tb[-c(1, length(tb$p)),]
ordered_tb <- tb[order(tb$p),]
top_n <- rownames(ordered_tb[1:n,])
top_n

train <- train[, top_n]
train$occurs <- occurs_train
train$los <- los_train

# Fitting model with selected features
(fit <- survreg(Surv(los,occurs) ~ . -los -occurs, data = train, dist=dist))
# n <- names(train)
# f <- as.formula(paste("Surv(train$los,train$occurs) ~", paste("train$",n[!n %in% "train"], collapse = " + "), "-train$los -train$occurs"))
# (fit <- tbs.survreg.be(f))
# coefs <- data.matrix(fit$beta)


# Now selecting same variables for val and test
val <- subset(val, select=top_n)
val$occurs <- occurs_val
val$los <- los_val

test <- subset(test, select=top_n)
test$occurs <- occurs_test
test$los <- los_test

# Plotting prediction versus actual
y_pred <- predict(fit, val)
# y_pred <- data.matrix(val) %*% coefs
y_val <- val$los
rmse <- sqrt(sum((y_pred-y_val)^2)/length(y_pred))

rmse
plot(y_val, y_pred,
     main="Predicted vs. True LOS on Val Set", 
     xlab="True LOS (days)", 
     ylab="Predicted LOS (days)",
     xlim=c(0,50), ylim=c(0,50))
abline(coef=c(0,1))

# Testing
y_pred <- predict(fit, test)
y_test <- test$los
rmse <- sqrt(sum((y_pred-y_test)^2)/length(y_pred))

rmse
plot(y_test, y_pred,
     main="Predicted vs. True LOS on Test Set", 
     xlab="True LOS (days)", 
     ylab="Predicted LOS (days)",
     xlim=c(0,50), ylim=c(0,50))
abline(coef=c(0,1))


# --------- Adding interaction terms ------------------ #

# Trying all possible interaction terms between selected features and selecting top k
(fit <- survreg(Surv(los,occurs) ~ (. -los -occurs)^2+(. -los -occurs)^3, data = train, dist=dist))

# Getting p-values for each feature and choosing top k
k=1
tb <- data.frame(summary(fit)$table)
tb <- tb[-c(1, length(tb$p)),]
ordered_tb <- tb[order(tb$p),]
top_k <- rownames(ordered_tb[1:k,])
top_k

# Doing cross-validation
val_rmses <- c()
test_rmses <- c()
best_val_rmse <- Inf
best_val_ypred <- NULL
best_yval <- NULL
best_test_ypred <- NULL
best_ytest <- NULL
set.seed(NULL)
for (i in 1:20) {
  
  all_data <- bind_rows(list(train, val, test))
  
  # Randomly splitting all_data into train, val, test
  train_ind <- sample(seq_len(nrow(los)), size = smp_size)
  train <- (all_data[train_ind, ])
  test <- (all_data[-train_ind, ])
  val_size <- floor(0.25 * nrow(train))
  val_ind <- sample(seq_len(nrow(train)), size = val_size)
  val <- train[val_ind,]
  train <- train[-val_ind,]
  
  # Fitting model with selected features
  f <- paste("Surv(los, occurs) ~ . -los -occurs+",(paste(top_k, collapse = '+')), sep="")
  # (fit <- survreg(Surv(los,occurs) ~ . -los -occurs+noquote(paste(top_k, collapse = '+')), data = train, dist=dist))
  (fit <- do.call("survreg", list(as.formula(f), data=as.name("train"), dist=as.name("dist"))))
  
  # Plotting prediction versus actual
  y_pred <- predict(fit, val)
  y_val <- val$los
  rmse <- sqrt(sum((y_pred-y_val)^2)/length(y_pred))
  
  rmse
  val_rmses <- c(val_rmses, rmse)

  plot(y_val, y_pred,
       main="Predicted vs. True LOS on Val Set",
       xlab="True LOS (days)",
       ylab="Predicted LOS (days)",
       xlim=c(0,50), ylim=c(0,50))
  abline(coef=c(0,1))
  
  # Testing
  y_pred1 <- predict(fit, test)
  y_test <- test$los
  rmse1 <- sqrt(sum((y_pred1-y_test)^2)/length(y_pred1))
  
  rmse1
  test_rmses <- c(test_rmses, rmse1)
  if (rmse < best_val_rmse) {
    best_val_rmse <- rmse
    best_val_ypred <- y_pred
    best_yval <- y_val
    best_test_ypred <- y_pred1
    best_ytest <- y_test
  }
  plot(y_test, y_pred1,
       main="Predicted vs. True LOS on Test Set",
       xlab="True LOS (days)",
       ylab="Predicted LOS (days)",
       xlim=c(0,50), ylim=c(0,50))
  abline(coef=c(0,1))
}

val_rmses
test_rmses

mean(val_rmses)
sd(val_rmses)
mean(test_rmses)
sd(test_rmses)

plot(best_yval, best_val_ypred,
     main="Predicted vs. True LOS on Val Set",
     xlab="True LOS (days)",
     ylab="Predicted LOS (days)",
     xlim=c(0,50), ylim=c(0,50))
abline(coef=c(0,1))

plot(best_ytest, best_test_ypred,
     main="Predicted vs. True LOS on Test Set",
     xlab="True LOS (days)",
     ylab="Predicted LOS (days)",
     xlim=c(0,50), ylim=c(0,50))
abline(coef=c(0,1))