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

preproc_center_scale <- preProcess(data, method=c("center", "scale"))
normalized_data <- predict(preproc_center_scale, data)
# Dados normalizados com média centralizada em 0

set.seed(202493)
ind <- createDataPartition(normalized_data$tipo, p=0.8, list=F)
train <- normalized_data[ind,]
test <- normalized_data[-ind,]

# --- Hold out ---

set.seed(202493)
rna <- train(form= tipo ~ ., data=train, method="nnet", tuneGrid=tune_grid, maxit=2000, trace=FALSE)
rna
# size = 5, decay = 0.1

predict.rna <- predict(rna, test)
confusionMatrix(predict.rna, as.factor(test$tipo))
# Accuracy: 0.8503

# --- Cross Validation ---

set.seed(202493)
train_control <- trainControl(method = "cv", number = 10)
rna <- train(form= tipo ~ ., data=train, method="nnet", trControl=train_control, maxit=2000, trace=FALSE)
rna
# size = 5, decay = 0.1

predict.rna <- predict(rna, test)
confusionMatrix(predict.rna, as.factor(test$tipo))
# Accuracy: 0.8024

# --- Cross Validation ---

tune_grid <- expand.grid(size = seq(from = 1, to = 45, by = 5), decay = seq(from = 0.1, to = 0.9, by = 0.3))
set.seed(202493)
train_control <- trainControl(method = "cv", number = 10)
rna <- train(form= tipo ~ ., data=train, method="nnet", trControl=train_control, tuneGrid=tune_grid, maxit=2000, trace=FALSE)
rna
# size = 11, decay = 0.4

predict.rna <- predict(rna, test)
confusionMatrix(predict.rna, as.factor(test$tipo))
# Accuracy: 0.8084

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
ind <- createDataPartition(normalized_data$diabetes, p=0.8, list=FALSE)
train <- normalized_data[ind,]
test <- normalized_data[-ind,]

# --- Hold out ---

set.seed(202493)
rna <- train(form= diabetes ~ ., data=train, method="nnet", maxit=2000, trace=FALSE)
rna
# size = 6, decay = 0.7

predict.rna <- predict(rna, test)
confusionMatrix(predict.rna, as.factor(test$diabetes))
# Accuracy: 0.7778

# --- Cross Validation ---

set.seed(202493)
train_control <- trainControl(method = "cv", number = 10)
rna <- train(form= diabetes ~ ., data=train, method="nnet", trControl=train_control, maxit=2000, trace=FALSE)
rna
# size = 3, decay = 0.1

predict.rna <- predict(rna, test)
confusionMatrix(predict.rna, as.factor(test$diabetes))
# Accuracy: 0.7451

# --- Cross Validation com tune grid ---

tune_grid <- expand.grid(size = seq(from = 1, to = 45, by = 5), decay = seq(from = 0.1, to = 0.9, by = 0.3))
set.seed(202493)
train_control <- trainControl(method = "cv", number = 10)
rna <- train(form= diabetes ~ ., data=train, method="nnet", trControl=train_control, tuneGrid=tune_grid, maxit=2000, trace=FALSE)
rna
# size = 11, decay = 0.7

predict.rna <- predict(rna, test)
confusionMatrix(predict.rna, as.factor(test$diabetes))
# Accuracy: 0.7451

# ------------------------------------------------

# 2.a admissão (regressão)
install.packages("e1071")
install.packages("caret")
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
rna <- train(form=ChanceOfAdmit ~ ., data=train, method="nnet", tuneGrid=tune_grid, linout=T, MaxNWts=10000, maxit=2000, trace=F)
rna
# size = 41, decay = 0.1

predict.rna <- predict(rna, test)

r2 <- function(predicted, observed) {
  return (1 - (sum((predicted - observed) ^ 2) / sum((observed - mean(observed)) ^ 2)))
}

syx <- function(predicted, observed) {
  n <- length(observed)
  syx <- sqrt(sum((observed - predicted)^2) / (n - 2))
  return(syx)
}

rmse(test$ChanceOfAdmit, predict.rna)
# 0.05834788

r2(predict.rna, test$ChanceOfAdmit)
# 0.8340659

syx(predict.rna, test$ChanceOfAdmit)
# 0.05895254

cor(test$ChanceOfAdmit, predict.rna) # Pearson (library stats)
# 0.9135127

mae(test$ChanceOfAdmit, predict.rna)
# 0.04407469

# --- Cross Validation ---

set.seed(202493)
train_control <- trainControl(method = "cv", number = 10)
rna <- train(form=ChanceOfAdmit ~ ., data=train, method="nnet", trControl=train_control, maxit=2000, trace=FALSE)
rna
# size = 1, decay = 0,0001

predict.rna <- predict(rna, test)

rmse(test$ChanceOfAdmit, predict.rna)
# 0.05978618

r2(predict.rna, test$ChanceOfAdmit)
# 0.8257844

syx(predict.rna, test$ChanceOfAdmit)
# 0.06040574

cor(test$ChanceOfAdmit, predict.rna) # Pearson (library stats)
# 0.9091133

mae(test$ChanceOfAdmit, predict.rna)
# 0.04494378

# --- Cross Validation com tune grid ---

tune_grid <- expand.grid(size = seq(from = 1, to = 45, by = 5), decay = seq(from = 0.1, to = 0.9, by = 0.3))
set.seed(202493)
train_control <- trainControl(method = "cv", number = 10)
rna <- train(form=ChanceOfAdmit ~ ., data=train, method="nnet", trControl=train_control, tuneGrid=tune_grid, maxit=2000, trace=FALSE)
rna
# size = 16, decay = 0.1

predict.rna <- predict(rna, test)

rmse(test$ChanceOfAdmit, predict.rna)
# 0.06047379

r2(predict.rna, test$ChanceOfAdmit)
# 0.821754

syx(predict.rna, test$ChanceOfAdmit)
# 0.06110048

cor(test$ChanceOfAdmit, predict.rna) # Pearson (library stats)
# 0.9093754

mae(test$ChanceOfAdmit, predict.rna)

# ------------------------------------------------

# 2.b biomassa (regressão)
install.packages("e1071")
install.packages("caret")
library("caret")
library(Metrics)
library(stats)

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
rna <- train(form=biomassa ~ ., data=train, method="nnet", tuneGrid=tune_grid, linout=T, MaxNWts=10000, maxit=2000, trace=F)
rna
# size = 1, decay = 0.4

predict.rna <- predict(rna, test)

r2 <- function(predicted, observed) {
  return (1 - (sum((predicted - observed) ^ 2) / sum((observed - mean(observed)) ^ 2)))
}

syx <- function(predicted, observed) {
  n <- length(observed)
  syx <- sqrt(sum((observed - predicted)^2) / (n - 2))
  return(syx)
}

rmse(test$biomassa, predict.rna)
# 321.6455

r2(predict.rna, test$biomassa)
# 0.8645137

syx(predict.rna, test$biomassa)
# 327.1441

cor(test$biomassa, predict.rna) # Pearson (library stats)
# 0.9753561

mae(test$biomassa, predict.rna)
# 165.1712

# ---  Cross validation ---

set.seed(202493)
train_control <- trainControl(method = "cv", number = 10)
rna <- train(form=biomassa ~ ., data=train, method="nnet", trControl=train_control, linout=T, maxit=2000, trace=FALSE)
rna
# size = 5, decay = 0.1

predict.rna <- predict(rna, test)

rmse(test$biomassa, predict.rna)
# 742.978

r2(predict.rna, test$biomassa)
# 0.2770764

syx(predict.rna, test$biomassa)
# 755.6794

cor(test$biomassa, predict.rna) 
# 0.9607058

mae(test$biomassa, predict.rna)
# 212.6646

# ---  Cross validation com tune grid ---

tune_grid <- expand.grid(size = seq(from = 1, to = 45, by = 5), decay = seq(from = 0.1, to = 0.9, by = 0.2))
set.seed(202493)
train_control <- trainControl(method = "cv", number = 10)
rna <- train(form=biomassa ~ ., data=train, method="nnet", trControl=train_control, tuneGrid=tune_grid, linout=T, maxit=2000, trace=FALSE)
rna
# size = 36, decay = 0.1

predict.rna <- predict(rna, test)

rmse(test$biomassa, predict.rna)
# 214.5112

r2(predict.rna, test$biomassa)
# 0.9397384

syx(predict.rna, test$biomassa)
# 218.1783

cor(test$biomassa, predict.rna) 
# 0.9767913  

mae(test$biomassa, predict.rna)
# 106.5668