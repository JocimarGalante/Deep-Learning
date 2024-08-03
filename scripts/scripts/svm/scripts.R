# 1.a veículos (classificação)
install.packages("e1071")
install.packages("caret")
library("caret")

setwd("/Users/cassi/dev/_estudos/pos-iaa/IAA008-aprendizado-maquina/bases-de-dados/06-veículos")

data <- read.csv("6-veiculos.csv")
View(data)

data$a <- NULL

any(is.na(data))
# FALSE

preproc_center_scale <- preProcess(data, method = c("center", "scale"))
normalized_data <- predict(preproc_center_scale, data)
# Dados normalizados com média centralizada em 0

set.seed(202493)
ind <- createDataPartition(normalized_data$tipo, p = 0.8, list = F)
train <- normalized_data[ind,]
test <- normalized_data[-ind,]

# --- Hold out ---

set.seed(202493)
svm <- train(tipo ~ ., data = normalized_data, method = "svmRadial")
svm
# sigma = 0.07189928; C = 1.

predict.svm <- predict(svm, test)
confusionMatrix(predict.svm, as.factor(test$tipo))
# Accuracy: 0.8503

# --- Cross Validation ---
set.seed(202493)
train_control <- trainControl(method = "cv", number = 10)
svm <- train(form= tipo ~ ., data=train, method="svmRadial", trControl=train_control)
svm
# sigma = 0.06598008; C = 1

predict.svm <- predict(svm, test)
confusionMatrix(predict.svm, as.factor(test$tipo))
# Accuracy: 0.7365

# --- Cross Validation com tune grid ---
tune_grid <- expand.grid(C = c(1, 2, 10, 50, 100), sigma = c(0.01, 0.015, 0.2))
set.seed(202493)
train_control <- trainControl(method = "cv", number = 10)
svm <- train(form= tipo ~ ., data=train, method="svmRadial", trControl=train_control, tuneGrid = tune_grid)
svm
# sigma = 0.015; C = 100

predict.svm <- predict(svm, test)
confusionMatrix(predict.svm, as.factor(test$tipo))
# Accuracy: 0.8443

# ------------------------------------------------

# 1.b diabetes (classificação)
install.packages("e1071")
install.packages("caret")
library("caret")

setwd("/Users/cassi/dev/_estudos/pos-iaa/IAA008-aprendizado-maquina/bases-de-dados/10-diabetes")

data <- read.csv("10-diabetes.csv")
View(data)

data$num <- NULL

any(is.na(data))
# FALSE

preproc_center_scale <- preProcess(data, method=c("center", "scale"))
normalized_data <- predict(preproc_center_scale, data)
# Dados normalizados com média centralizada em 0

set.seed(202493)
ind <- createDataPartition(normalized_data$diabetes, p = 0.8, list = FALSE)
train <- normalized_data[ind,]
test <- normalized_data[-ind,]

# --- Hold out ---

set.seed(202493)
svm <- train(diabetes ~ ., data = normalized_data, method = "svmRadial")
svm
# sigma = 0.1258432; C = 0.25.

predict.svm <- predict(svm, test)
confusionMatrix(predict.svm, as.factor(test$diabetes))
# Accuracy: 0.8503

# --- Cross Validation ---

set.seed(202493)
train_control <- trainControl(method = "cv", number = 10)
svm <- train(form = diabetes ~ ., data=train, method="svmRadial", trControl=train_control)
svm
# sigma = 0.1329232; C = 0.25

predict.svm <- predict(svm, test)
confusionMatrix(predict.svm, as.factor(test$diabetes))
# Accuracy: 0.7712

# --- Cross Validation com tune grid ---

tune_grid <- expand.grid(C = c(1, 2, 10, 50, 100), sigma = c(0.01, 0.015, 0.2))
set.seed(202493)
train_control <- trainControl(method = "cv", number = 10)
svm <- train(form = diabetes ~ ., data=train, method="svmRadial", trControl=train_control, tuneGrid = tune_grid)
svm
# sigma = 0.015; C = 2

predict.svm <- predict(svm, test)
confusionMatrix(predict.svm, as.factor(test$diabetes))
# Accuracy: 0.7712

# ------------------------------------------------

# 2.a admissão (regressão)
install.packages("e1071")
install.packages("caret")
install.packages("kernlab")
install.packages("mlbench")
install.packages("mice")
library(mlbench)
library("caret")
library(Metrics)
library(stats)
library(mice)

setwd("/Users/cassi/dev/_estudos/pos-iaa/IAA008-aprendizado-maquina/bases-de-dados/09-admissão")

data <- read.csv("9-admissao.csv")
View(data)

data$num <- NULL

any(is.na(data))
# FALSE

target_data <- data[["ChanceOfAdmit"]]
predictors <- data[, colnames(data) != "ChanceOfAdmit"]

preproc_center_scale <- preProcess(predictors, method=c("center", "scale"))
normalized_predictors <- predict(preproc_center_scale, predictors)

normalized_data <- cbind(normalized_predictors, target_data)

names(normalized_data)[names(normalized_data) == "target_data"] <- "ChanceOfAdmit"

set.seed(202493)
ind <- createDataPartition(normalized_data$ChanceOfAdmit, p=0.8, list=FALSE)
train <- normalized_data[ind,]
test <- normalized_data[-ind,]

# --- Hold out ---

set.seed(202493)
svm <- train(form=ChanceOfAdmit ~ ., data=train, method="svmRadial")
svm
# sigma = 0.1769097, C = 0.5

predict.svm <- predict(svm, test)

r2 <- function(predicted, observed) {
  return (1 - (sum((predicted - observed) ^ 2) / sum((observed - mean(observed)) ^ 2)))
}

syx <- function(predicted, observed) {
  n <- length(observed)
  syx <- sqrt(sum((observed - predicted)^2) / (n - 2))
  return(syx)
}

rmse(test$ChanceOfAdmit, predict.svm)
# 0.06364335

r2(predict.svm, test$ChanceOfAdmit)
# 0.8025798

syx(predict.svm, test$ChanceOfAdmit)
# 0.06430289

cor(test$ChanceOfAdmit, predict.svm) # Pearson (library stats)
# 0.8979676

mae(test$ChanceOfAdmit, predict.svm)
# 0.04582943

# --- Cross Validation ---

set.seed(202493)
train_control <- trainControl(method = "cv", number = 10)
svm <- train(form= ChanceOfAdmit ~ ., data=train, method="svmRadial", trControl=train_control)
svm
# sigma = 0.1769097; C = 1

predict.svm <- predict(svm, test)

rmse(test$ChanceOfAdmit, predict.svm)
# 0.06339046

r2(predict.svm, test$ChanceOfAdmit)
# 0.8041456

syx(predict.svm, test$ChanceOfAdmit)
# 0.06404738

cor(test$ChanceOfAdmit, predict.svm) # Pearson (library stats)
# 0.8982254

mae(test$ChanceOfAdmit, predict.svm)
# 0.0455813


# --- Cross Validation com tune grid ---

tune_grid <- expand.grid(C = c(1, 2, 10, 50, 100), sigma = c(0.01, 0.015, 0.2))
set.seed(202493)
train_control <- trainControl(method = "cv", number = 10)
svm <- train(form= ChanceOfAdmit ~ ., data=train, method="svmRadial", trControl=train_control, tuneGrid = tune_grid)
svm
# sigma = 0.015; C = 50

predict.svm <- predict(svm, test)

rmse(test$ChanceOfAdmit, predict.svm)
# 0.06062261

r2(predict.svm, test$ChanceOfAdmit)
# 0.8208756

syx(predict.svm, test$ChanceOfAdmit)
# 0.06125084

cor(test$ChanceOfAdmit, predict.svm) # Pearson (library stats)
# 0.9091315

mae(test$ChanceOfAdmit, predict.svm)
# 0.04388186

# 2.b biomassa (regressão)
install.packages("e1071")
install.packages("caret")
install.packages("mlbench")
install.packages("mice")
library(mlbench)
library("caret")
library(Metrics)
library(stats)
library(mice)

setwd("/Users/cassi/dev/_estudos/pos-iaa/IAA008-aprendizado-maquina/bases-de-dados/05-biomassa")

data <- read.csv("5-biomassa.csv")
View(data)

any(is.na(data))
# FALSE

target_data <- data[["biomassa"]]
predictors <- data[, colnames(data) != "biomassa"]

preproc_center_scale <- preProcess(predictors, method=c("center", "scale"))
normalized_predictors <- predict(preproc_center_scale, predictors)

normalized_data <- cbind(normalized_predictors, target_data)

names(normalized_data)[names(normalized_data) == "target_data"] <- "biomassa"

set.seed(202493)
ind <- createDataPartition(normalized_data$biomassa, p=0.8, list=FALSE)
train <- normalized_data[ind,]
test <- normalized_data[-ind,]

# --- Hold out ---

set.seed(202493)
svm <- train(form=biomassa ~ ., data=train, method="svmRadial")
svm
# sigma = 1.443104, C = 1

predict.svm <- predict(svm, test)

r2 <- function(predicted, observed) {
  return (1 - (sum((predicted - observed) ^ 2) / sum((observed - mean(observed)) ^ 2)))
}

syx <- function(predicted, observed) {
  n <- length(observed)
  syx <- sqrt(sum((observed - predicted)^2) / (n - 2))
  return(syx)
}

rmse(test$biomassa, predict.svm)
# 390.429

r2(predict.svm, test$biomassa)
# 0.8003705

syx(predict.svm, test$biomassa)
# 397.1035

cor(test$biomassa, predict.svm) # Pearson (library stats)
# 0.9289524

mae(test$biomassa, predict.svm)
# 169.6209

# --- Cros Validation ---

set.seed(202493)
train_control <- trainControl(method = "cv", number = 10)
svm <- train(form= biomassa ~ ., data=train, method="svmRadial", trControl=train_control)
svm
# sigma = 1.254606; C = 100

predict.svm <- predict(svm, test)

rmse(test$biomassa, predict.svm)
# 358.9729

r2(predict.svm, test$biomassa)
# 0.8312422

syx(predict.svm, test$biomassa)
# 365.1096


cor(test$biomassa, predict.svm) # Pearson (library stats)
# 0.938767


mae(test$biomassa, predict.svm)
# 162.3039


# --- Cross Validation com tune grid ---

tune_grid <- expand.grid(C = c(1, 2, 10, 50, 100), sigma = c(0.01, 0.015, 0.2))
set.seed(202493)
train_control <- trainControl(method = "cv", number = 10)
svm <- train(form= biomassa ~ ., data=train, method="svmRadial", trControl=train_control, tuneGrid = tune_grid)
svm
# sigma = 0.01; C = 100

predict.svm <- predict(svm, test)

rmse(test$biomassa, predict.svm)
# 278.1665

r2(predict.svm, test$biomassa)
# 0.8986672

syx(predict.svm, test$biomassa)
# 282.9218

cor(test$biomassa, predict.svm) # Pearson (library stats)
# 0.9878348

mae(test$biomassa, predict.svm)
# 124.8422