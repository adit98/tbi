# Set working directory
setwd("~/Johns Hopkins University/2019-20/Fall 2019/Precision Care Medicine I/TBI_Project/tbi/R_survival_analysis")

library(tidyverse)
library(knitr)
library(ciTools)
library(here)
library(survival)
library(standardize)
library(BART)

set.seed(20180925)

#------------------- mort ----------------------#

# Loading data
mort = read.table("../notebooks/mort_surv_analysis_data.csv", sep=",", header=TRUE)
mort$GCS <- NULL
mort$Value <- NULL
mort$patientunitstayid <- NULL
mort$X <- NULL

# Displaying first entries in table
# kable(head(mort))

ggplot(mort, aes(x = DEM_age, y = los)) +
  geom_point(aes(color = factor(death)))+
  ggtitle("Censored obs. in red") +
  theme_bw()

# Splitting data into train and test
smp_size <- floor(0.75 * nrow(mort))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(mort)), size = smp_size)

train <- mort[train_ind, ]
test <- mort[-train_ind, ]

# los_train <- train$los
# death_train <- train$death
# los_test <- test$los
# death_test <- test$death
# 
# 
# train <- data.frame(scale(train))
# test <- data.frame(scale(test))
# 
# train$los <- los_train
# train$death <- death_train
# test$los <- los_test
# test$death <- death_test

dist="exponential"

# Using no covariates
(fit <- survreg(Surv(los,death) ~ 1, data = train, dist=dist)) ## weibull dist is default

# Using all covariates
(fit <- survreg(Surv(los,death) ~ . -los -death, data = train, dist=dist)) ## weibull dist is default

y_pred <- predict(fit, test)
y_test <- test$los
mse <- sum((y_pred-y_test)^2)/length(y_pred)

mse
plot(y_test, y_pred,
     main="Predicted vs. True LOS", 
     xlab="True LOS (days)", 
     ylab="Predicted LOS (days)")
     # xlim=c(0,50), ylim=c(0,50))

# Using mort coefficients
(fit <- survreg(Surv(los,death) ~ DEM_age+gcs0+INF_norepinephrine+INF_morphine+LAB_BUN+LAB_glucose+sao20+INF_phenylephrine+MED_4846.0+LAB_paCO2, data = train, dist=dist)) ## weibull dist is default

# Using LOS coefficients
(fit <- survreg(Surv(los,death) ~ gcs0+MED_549.0+MED_1326.0+MED_25386.0+LAB_HCO3+LAB_Hct+LAB_Hgb, data = train, dist=dist)) ## weibull dist is default




#------------------- LOS ----------------------#

# Loading data
los = read.table("../notebooks/los_surv_analysis_dat.csv", sep=",", header=TRUE)
los$GCS <- NULL
los$Value <- NULL
los$patientunitstayid <- NULL
los$X <- NULL

ggplot(los, aes(x = DEM_age, y = los)) +
  geom_point(aes(color = factor(occurs)))+
  ggtitle("Censored obs. in red") +
  theme_bw()

# Splitting data into train and test
smp_size <- floor(0.75 * nrow(los))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(los)), size = smp_size)

# train <- data.frame(scale(los[train_ind, ]))
# test <- data.frame(scale(los[-train_ind, ]))

train <- (los[train_ind, ])
test <- (los[-train_ind, ])

los_train <- train$los
occurs_train <- train$occurs
los_test <- test$los
occurs_test <- test$occurs


# train <- data.frame(scale(train))
# test <- data.frame(scale(test))
# 
# train$los <- los_train
# train$occurs <- occurs_train
# test$los <- los_test
# test$occurs <- occurs_test

train_norm <- as.data.frame(apply(train, 2, function(x) (x - min(x))/(max(x)-min(x))))
test_norm <- as.data.frame(apply(test, 2, function(x) (x - min(x))/(max(x)-min(x))))

train_norm$los <- los_train
train_norm$occurs <- occurs_train
test_norm$los <- los_test
test_norm$occurs <- occurs_test

train <- train_norm
test <- test_norm

dist="loglogistic"
# USing no covariates
(fit <- survreg(Surv(los,occurs) ~ 1, data = train, dist=dist)) ## weibull dist is default

# Using all covariates
(fit <- survreg(Surv(los,occurs) ~ . -los -occurs, data = train, dist=dist)) ## weibull dist is default

# Using mort coefficients (old)
(fit <- survreg(Surv(los,occurs) ~ DEM_age+gcs0+INF_norepinephrine+INF_morphine+LAB_BUN+LAB_glucose+sao20+INF_phenylephrine+MED_4846.0+LAB_paCO2, data = train, dist=dist)) ## weibull dist is default

# Using mort coefficients (new)
(fit <- survreg(Surv(los,occurs) ~ gcs0+DEM_age+MED_4846.0+INF_morphine+INF_norepinephrine+MED_1730.0+sao20+LAB_glucose+MED_1694.0+MED_4521.0, data = train, dist=dist)) ## weibull dist is default

# Using LOS coefficients
(fit <- survreg(Surv(los,occurs) ~ gcs0+MED_549.0+MED_1326.0+MED_25386.0+LAB_HCO3+LAB_Hct+LAB_Hgb, data = train, dist=dist)) ## weibull dist is default


# Using variable selection
x <- train
# x$los <- NULL
x$occurs <- NULL
model_lm <- lm(los ~ ., data=x)
fit1_lm <- stepAIC(model_lm, direction = 'backward')

train <- fit1_lm$model
train$occurs <- occurs_train
selected_vars <- names(train)

# Now selecting same variables for test
test <- subset(test, select=selected_vars)


# Plotting prediction versus actual
y_pred <- predict(fit, test)
y_test <- test$los
rmse <- sqrt(sum((y_pred-y_test)^2)/length(y_pred))

rmse
plot(y_test, y_pred,
     main="Predicted vs. True LOS", 
     xlab="True LOS (days)", 
     ylab="Predicted LOS (days)",
     xlim=c(0,50), ylim=c(0,50))
abline(coef=c(0,1))