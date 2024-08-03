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
tune_grid <- expand.grid(k=c(1, 3, 5, 7, 9))

set.seed(202493)
knn <- train(tipo ~ ., data=train, method="knn", tuneGrid=tune_grid)
knn
# k = 1

predict.knn <- predict(knn, test)
confusionMatrix(predict.knn, as.factor(test$tipo))
# Accuracy: 0.6766

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
tune_grid <- expand.grid(k=c(1, 3, 5, 7, 9))

set.seed(202493)
knn <- train(diabets ~ ., data=train, method="nkk", tuneGrid=tune_grid)
knn
# k = 9

predict.knn <- predict(knn, test)
confusionMatrix(predict.knn, as.factor(test$diabetes))
# Accuracy: 0.7255

# ------------------------------------------------

# 2.a admissão (regressão)
install.packages("e1071")
install.packages("caret")
library("caret")
library(Metrics)
library(stats)

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
tune_grid <- expand.grid(k=c(1, 3, 5, 7, 9))

set.seed(202493)
knn <- train(ChanceOfAdmit ~ ., data=train, method="knn", tuneGrid=tune_grid)
knn
# k = 9

predict.knn <- predict(knn, test)

r2 <- function(predicted, observed) {
  return (1 - (sum((predicted - observed) ^ 2) / sum((observed - mean(observed)) ^ 2)))
}

syx <- function(predicted, observed) {
  n <- length(observed)
  syx <- sqrt(sum((observed - predicted)^2) / (n - 2))
  return(syx)
}

rmse(test$ChanceOfAdmit, predict.knn)
# 0.065908

r2(predict.knn, test$ChanceOfAdmit)
# 0.7883

syx(predict.knn, test$ChanceOfAdmit)
# 0.06659101

cor(test$ChanceOfAdmit, predict.knn) # Pearson (library stats)
# 0.89068

mae(test$ChanceOfAdmit, predict.knn)
# 0.04750567

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
tune_grid <- expand.grid(k=c(1, 3, 5, 7, 9))

set.seed(202493)
knn <- train(biomassa ~ ., data=train, method="knn", tuneGrid=tune_grid)
knn
# k = 1

predict.knn <- predict(knn, test)

r2 <- function(predicted, observed) {
  return (1 - (sum((predicted - observed) ^ 2) / sum((observed - mean(observed)) ^ 2)))
}

syx <- function(predicted, observed) {
  n <- length(observed)
  syx <- sqrt(sum((observed - predicted)^2) / (n - 2))
  return(syx)
}

rmse(test$biomassa, predict.knn)
# 250.1385

r2(predict.knn, test$biomassa)
# 0.9181

syx(predict.knn, test$biomassa)
# 254.4146

cor(test$biomassa, predict.knn) # Pearson (library stats)
# 0.95862

mae(test$biomassa, predict.knn)
# 101.6077