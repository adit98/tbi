# Set working directory
setwd("~/Johns Hopkins University/2019-20/Fall 2019/Precision Care Medicine I/TBI_Project/tbi/R_survival_analysis")

library(tidyverse)
library(knitr)
library(ciTools)
library(here)
library(survival)

set.seed(20180925)

#------------------- mort ----------------------#

# Loading data
mort = read.table("../notebooks/mort_surv_analysis_data.csv", sep=",", header=TRUE)


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

dist="weibull"

# Using no covariates
(fit <- survreg(Surv(los,death) ~ 1, data = train, dist=dist)) ## weibull dist is default

# Using all covariates
(fit <- survreg(Surv(los,death) ~ . -los -death, data = train, dist=dist)) ## weibull dist is default

y_pred <- predict(fit, test)
y_test <- test$los
mse <- sum((y_pred-y_test)^2)/length(y_pred)

mse
plot(y_test, y_pred)

# Using mort coefficients
(fit <- survreg(Surv(los,death) ~ DEM_age+gcs0+INF_norepinephrine+INF_morphine+LAB_BUN+LAB_glucose+sao20+INF_phenylephrine+MED_4846.0+LAB_paCO2, data = train, dist=dist)) ## weibull dist is default

# Using LOS coefficients
(fit <- survreg(Surv(los,death) ~ gcs0+MED_549.0+MED_1326.0+MED_25386.0+LAB_HCO3+LAB_Hct+LAB_Hgb, data = train, dist=dist)) ## weibull dist is default




#------------------- LOS ----------------------#

# Loading data
los = read.table("../notebooks/los_surv_analysis_dat.csv", sep=",", header=TRUE)

ggplot(los, aes(x = DEM_age, y = los)) +
  geom_point(aes(color = factor(occurs)))+
  ggtitle("Censored obs. in red") +
  theme_bw()

# Splitting data into train and test
smp_size <- floor(0.75 * nrow(los))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(los)), size = smp_size)

train <- los[train_ind, ]
test <- los[-train_ind, ]

# USing no covariates
(fit <- survreg(Surv(los,occurs) ~ 1, data = train, dist=dist)) ## weibull dist is default

# Using all covariates
(fit <- survreg(Surv(los,occurs) ~ . -los -occurs, data = train, dist=dist)) ## weibull dist is default

y_pred <- predict(fit, test)
y_test <- test$los
mse <- sum((y_pred-y_test)^2)/length(y_pred)

mse
plot(y_pred, y_test)

# Using mort coefficients
(fit <- survreg(Surv(los,occurs) ~ DEM_age+gcs0+INF_norepinephrine+INF_morphine+LAB_BUN+LAB_glucose+sao20+INF_phenylephrine+MED_4846.0+LAB_paCO2, data = train, dist=dist)) ## weibull dist is default

# Using LOS coefficients
(fit <- survreg(Surv(los,occurs) ~ gcs0+MED_549.0+MED_1326.0+MED_25386.0+LAB_HCO3+LAB_Hct+LAB_Hgb, data = train, dist=dist)) ## weibull dist is default
