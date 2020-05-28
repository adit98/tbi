# Alternate way to select variables - doesn't work very well

x <- train
x$occurs <- NULL
x$los <- NULL

# INF_fentanyl is the same value for all rows in train
x$INF_fentanyl <- NULL
# # model_lm <- lm(los ~ ., data=x)
# model_lm <- survreg(Surv(los,occurs) ~ . -los -occurs, data = train, dist=dist)
# fit1_lm <- stepAIC(model_lm, direction = 'backward')
# 
# train <- fit1_lm$model
# train$occurs <- occurs_train
# selected_vars <- names(train)

# names. <- names(train)
# remove <- c("occurs", "los")
# names.[! names. %in% remove]
# 
# X <- as.matrix(train)[, names.]
# vars=srstepwise(X, train$los, train$occurs,0.45,0.2,dist='loglogistic')
# print(names.[vars])

# train <- train[, names.[vars]]
# train$occurs <- occurs_train
# train$los <- los_train
# 
# 
# # Now selecting same variables for test
# test <- subset(test, select=names.[vars])
# test$occurs <- occurs_test
# test$los <- los_test