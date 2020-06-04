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
nomed <- TRUE
nolab <- TRUE
rfefunc <- 'treebagFuncs' # Use rfFuncs or treebagFuncs

# Loading data
los = read.table("../notebooks/los_surv_analysis_dat.csv", sep=",", header=TRUE)
los$GCS <- NULL
los$Value <- NULL
los$patientunitstayid <- NULL
los$X <- NULL

los <- los[los['los'] <= 30,]

# Removing columns that start with MED or LAB
if (nomed) {
  los <- los[,!startsWith(colnames(los), 'MED')]
}
if (nolab) {
  los <- los[,!startsWith(colnames(los), 'LAB')]
}

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

dist="loglogistic"

nfeat <- 30
cols <- NULL
# ------------------ RFE ----------------- #
if (rfefunc == 'rfFuncs') {
  ctrl <- rfeControl(functions = rfFuncs,
                     method = "repeatedcv",
                     repeats = 5,
                     verbose = FALSE)
  x <- train[, !colnames(train) %in% c("los", "occurs")]
  y <- train$los
  # lmProfile <- rfe(x,y,rfeControl=ctrl)
  cols <- rownames(lmProfile$fit$importance)[0:nfeat] # Use with rfFUncs
} else if (rfefunc == 'treebagFuncs') {
  ctrl <- rfeControl(functions = treebagFuncs,
                     method = "repeatedcv",
                     repeats = 5,
                     verbose = FALSE)
  x <- train[, !colnames(train) %in% c("los", "occurs")]
  y <- train$los
  # lmProfile <- rfe(x,y,rfeControl=ctrl)
  cols <- colnames((lmProfile$fit$X)[0:nfeat]) # Use with treebagFuncs
}
cols

# Getting selected columns in train data
train <- train[, cols]
train$occurs <- occurs_train
train$los <- los_train

# Now selecting same variables for val and test
val <- subset(val, select=cols)
val$occurs <- occurs_val
val$los <- los_val

test <- subset(test, select=cols)
test$occurs <- occurs_test
test$los <- los_test

# ----------------- END RFE ---------------- #

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
  (fit <- survreg(Surv(los,occurs) ~ . -los -occurs, data = train, dist=dist))
  
  # Plotting prediction versus actual
  y_pred <- predict(fit, val)
  # y_pred <- data.matrix(val) %*% coefs
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