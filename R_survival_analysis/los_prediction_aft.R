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

# Loading data
los = read.table("../notebooks/los_surv_analysis_dat.csv", sep=",", header=TRUE)
los$GCS <- NULL
los$Value <- NULL
los$patientunitstayid <- NULL
los$X <- NULL

los <- los[los['los'] <= 30,]

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

dist="loglogistic"

# Using variable selection

# Fitting model
(fit <- survreg(Surv(los,occurs) ~ . -los -occurs, data = train, dist=dist))
# n <- names(train)
# f <- as.formula(paste("Surv(train$los,train$occurs) ~", paste("train$",n[!n %in% "train"], collapse = " + "), "-train$los -train$occurs"))
# (fit <- tbs.survreg.be(f))

# Getting p-values for each feature and choosing top n
n=12 # Best: 9 with normalization (los <= 30) 14 with normalization, 19 without (all los)
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

# Fitting model with selected features
f <- paste("Surv(los, occurs) ~ . -los -occurs+",(paste(top_k, collapse = '+')), sep="")
# (fit <- survreg(Surv(los,occurs) ~ . -los -occurs+noquote(paste(top_k, collapse = '+')), data = train, dist=dist))
(fit <- do.call("survreg", list(as.formula(f), data=as.name("train"), dist=as.name("dist"))))

# Plotting prediction versus actual
y_pred <- predict(fit, val)
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