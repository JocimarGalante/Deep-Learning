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

View(normalized_data)

set.seed(202493)
ind <- createDataPartition(normalized_data$tipo, p = 0.8, list = F)
train <- normalized_data[ind,]
test <- normalized_data[-ind,]

# --- Hold out ---

set.seed(202493)
rf <- train(tipo ~ ., data = normalized_data, method = "rf")
rf
# mtry = 2

predict.rf <- predict(rf, test)
confusionMatrix(predict.rf, as.factor(test$tipo))
# Accuracy: 1

# --- Cross Validation ---

set.seed(202493)
train_control <- trainControl(method = "cv", number = 10)
rf <- train(form = tipo ~ ., data=train, method="rf", trControl=train_control)
rf
# mtry = 10

predict.rf <- predict(rf, test)
confusionMatrix(predict.rf, as.factor(test$tipo))
# Accuracy: 0.7365

# --- Cross Validation com tune grid ---

tune_grid <- expand.grid(mtry = c(2, 5, 7, 9, 11, 13, 15, 17, 19, 21))
set.seed(202493)
train_control <- trainControl(method = "cv", number = 10)
rf <- train(form = tipo ~ ., data=train, method="rf", trControl=train_control, tuneGrid = tune_grid)
rf
# mtry = 5

predict.rf <- predict(rf, test)
confusionMatrix(predict.rf, as.factor(test$tipo))
# Accuracy: 0.7305

# --- Novos casos (usando Hold out) ----

new_data <- read.csv("6-veiculos-novos-casos.csv")
View(new_data)

new_data$a <- NULL

any(is.na(new_data))
# FALSE

preproc_center_scale <- preProcess(new_data, method = c("center", "scale"))
normalized_new_data <- predict(preproc_center_scale, new_data)
# Dados normalizados com média centralizada em 0

View(normalized_new_data)

predict.rf_new_data <- predict(rf, normalized_new_data)
# van  bus  opel
# Levels: bus opel saab van

new_data$tipo <- NULL
result <- cbind(new_data, predict.rf_new_data)
names(result)[names(result) == "predict.rf_new_data"] <- "tipo"
View(result)
# Visualização do DF com os novos dados e a predição

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

View(normalized_data)

set.seed(202493)
ind <- createDataPartition(normalized_data$diabetes, p = 0.8, list = FALSE)
train <- normalized_data[ind,]
test <- normalized_data[-ind,]

# --- Hold out ---

set.seed(202493)
rf <- train(diabetes ~ ., data = normalized_data, method = "rf")
rf
# mtry = 2

predict.rf <- predict(rf, test)
confusionMatrix(predict.rf, as.factor(test$diabetes))
# Accuracy: 1

# --- Cross Validation ---

set.seed(202493)
train_control <- trainControl(method = "cv", number = 10)
rf <- train(form = diabetes ~ ., data=train, method="rf", trControl=train_control)
rf
# mtry = 2

predict.rf <- predict(rf, test)
confusionMatrix(predict.rf, as.factor(test$diabetes))
# Accuracy: 0.7712

# --- Cross Validation com tune grid ---

tune_grid <- expand.grid(mtry = c(1, 3, 5, 7, 9, 11, 13))
set.seed(202493)
train_control <- trainControl(method = "cv", number = 10)
rf <- train(form = diabetes ~ ., data=train, method="rf", trControl=train_control, tuneGrid = tune_grid)
rf
# mtry = 9

predict.rf <- predict(rf, test)
confusionMatrix(predict.rf, as.factor(test$diabetes))
# Accuracy: 0.7451

# --- Novos casos (usando Hold out) ----

new_data <- read.csv("10-diabetes-novos-dados.csv")
View(new_data)

new_data$a <- NULL

any(is.na(new_data))
# FALSE

preproc_center_scale <- preProcess(new_data, method = c("center", "scale"))
normalized_new_data <- predict(preproc_center_scale, new_data)
# Dados normalizados com média centralizada em 0

View(normalized_new_data)

predict.rf_new_data <- predict(rf, normalized_new_data)
# pos neg neg 
# Levels: neg pos 

new_data$diabetes <- NULL
result <- cbind(new_data, predict.rf_new_data)
names(result)[names(result) == "predict.rf_new_data"] <- "diabetes"
View(result)
# Visualização do DF com os novos dados e a predição

# ------------------------------------------------

# 2.a admissão (regressão)
install.packages("e1071")
install.packages("kernlab")
install.packages("caret")
install.packages("mice")
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

View(normalized_data)

set.seed(202493)
ind <- createDataPartition(normalized_data$ChanceOfAdmit, p = 0.8, list = FALSE)
train <- normalized_data[ind,]
test <- normalized_data[-ind,]

# --- Hold out ---

set.seed(202493)
rf_ho <- train(ChanceOfAdmit ~ ., data = normalized_data, method = "rf")
rf_ho
# mtry = 2

predict.rf_ho <- predict(rf_ho, test)

r2 <- function(predicted, observed) {
  return (1 - (sum((predicted - observed) ^ 2) / sum((observed - mean(observed)) ^ 2)))
}

syx <- function(predicted, observed) {
  n <- length(observed)
  syx <- sqrt(sum((observed - predicted)^2) / (n - 2))
  return(syx)
}

rmse(test$ChanceOfAdmit, predict.rf_ho)
# 0.0333386

r2(predict.rf_ho, test$ChanceOfAdmit)
# 0.9458273

syx(predict.rf_ho, test$ChanceOfAdmit)
# 0.03368409

cor(test$ChanceOfAdmit, predict.rf_ho) # Pearson (library stats)
# 0.9746234

mae(test$ChanceOfAdmit, predict.rf_ho)
# 0.02295854

# --- Cross Validation ---

set.seed(202493)
train_control <- trainControl(method = "cv", number = 10)
rf <- train(form = ChanceOfAdmit ~ ., data=train, method="rf", trControl=train_control)
rf
# mtry = 2

predict.rf <- predict(rf, test)

rmse(test$ChanceOfAdmit, predict.rf)
# 0.06351048

r2(predict.rf, test$ChanceOfAdmit)
# 0.8034033

syx(predict.rf, test$ChanceOfAdmit)
# 0.06416863

cor(test$ChanceOfAdmit, predict.rf) # Pearson (library stats)
# 0.8966502

mae(test$ChanceOfAdmit, predict.rf)
# 0.04564234

# --- Cross Validation com tune grid ---

tune_grid <- expand.grid(mtry = c(2, 5, 7, 9))
set.seed(202493)
train_control <- trainControl(method = "cv", number = 10)
rf <- train(form = ChanceOfAdmit ~ ., data=train, method="rf", trControl=train_control, tuneGrid = tune_grid)
rf
# mtry = 2

predict.rf <- predict(rf, test)

rmse(test$ChanceOfAdmit, predict.rf)
# 0.06332055

r2(predict.rf, test$ChanceOfAdmit)
# 0.8045774

syx(predict.rf, test$ChanceOfAdmit)
# 0.06397674

cor(test$ChanceOfAdmit, predict.rf) # Pearson (library stats)
# 0.8973206

mae(test$ChanceOfAdmit, predict.rf)
# 0.04513112

# --- Novos casos (usando Hold out) ----

new_data <- read.csv("9-admissao-novos-dados.csv")
View(new_data)

new_data$num <- NULL

any(is.na(new_data))
# FALSE

new_target_data <- new_data[["ChanceOfAdmit"]]
new_predictors <- new_data[, colnames(new_data) != "ChanceOfAdmit"]

preproc_center_scale <- preProcess(new_predictors, method=c("center", "scale"))
normalized_new_predictors <- predict(preproc_center_scale, new_predictors)

normalized_new_data <- cbind(normalized_new_predictors, new_target_data)

names(normalized_new_data)[names(normalized_new_data) == "new_target_data"] <- "ChanceOfAdmit"
# Dados normalizados com média centralizada em 0

View(normalized_new_data)

predict.rf_ho_new_data <- predict(rf_ho, normalized_new_data)
predict.rf_ho_new_data
#         1         2         3
# 0.6088426 0.7209769 0.7601318

new_data$ChanceOfAdmit <- NULL
result <- cbind(new_data, predict.rf_ho_new_data)
names(result)[names(result) == "predict.rf_ho_new_data"] <- "ChanceOfAdmit"
View(result)
# Visualização do DF com os novos dados e a predição


# --- Geração do Gráfico de Resíduos com RF Hold Out e Dados de teste ---

test_residuals <- ((test$ChanceOfAdmit - predict.rf_ho) / test$ChanceOfAdmit) * 100

plot(predict.rf_ho, test_residuals, col = "blue", pch = 20, main = "Resíduos (%) - RF Hold Out (Dados teste)", xlab = "ChanceOfAdmit (estimado)", ylab = "Resíduo (%)", ylim=c(-100, 100))
abline(h = 0, col = "gray")
grid()

# ------------------------------------------------

# 2.b biomassa (regressão)
install.packages("e1071")
install.packages("kernlab")
install.packages("caret")
install.packages("mice")
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

View(normalized_data)

set.seed(202493)
ind <- createDataPartition(normalized_data$biomassa, p=0.8, list=FALSE)
train <- normalized_data[ind,]
test <- normalized_data[-ind,]

# --- Hold out ---

set.seed(202493)
rf_ho <- train(biomassa ~ ., data = normalized_data, method = "rf")
rf_ho
# mtry = 2

predict.rf_ho <- predict(rf_ho, test)

r2 <- function(predicted, observed) {
  return (1 - (sum((predicted - observed) ^ 2) / sum((observed - mean(observed)) ^ 2)))
}

syx <- function(predicted, observed) {
  n <- length(observed)
  syx <- sqrt(sum((observed - predicted)^2) / (n - 2))
  return(syx)
}

rmse(test$biomassa, predict.rf_ho)
# 123.4682

r2(predict.rf_ho, test$biomassa)
# 0.9800358

syx(predict.rf_ho, test$biomassa)
# 125.5789

cor(test$biomassa, predict.rf_ho) # Pearson (library stats)
# 0.9963851

mae(test$biomassa, predict.rf_ho)
# 41.88866

# --- Cross Validation ---

set.seed(202493)
train_control <- trainControl(method = "cv", number = 10)
rf <- train(form = biomassa ~ ., data=train, method="rf", trControl=train_control)
rf
# mtry = 2

predict.rf <- predict(rf, test)

rmse(test$biomassa, predict.rf)
# 158.1995

r2(predict.rf, test$biomassa)
# 0.9672244

syx(predict.rf, test$biomassa)
# 160.904

cor(test$biomassa, predict.rf) # Pearson (library stats)
# 0.9940073

mae(test$biomassa, predict.rf)
# 67.7821

# --- Cross Validation com tune grid ---

tune_grid <- expand.grid(mtry = c(1, 3, 5, 7, 9))
set.seed(202493)
train_control <- trainControl(method = "cv", number = 10)
rf <- train(form = biomassa ~ ., data=train, method="rf", trControl=train_control, tuneGrid = tune_grid)
rf
# mtry = 3

predict.rf <- predict(rf, test)

rmse(test$biomassa, predict.rf)
# 150.9076

r2(predict.rf, test$biomassa)
# 0.9701762

syx(predict.rf, test$biomassa)
# 153.4874

cor(test$biomassa, predict.rf) # Pearson (library stats)
# 0.9917952

mae(test$biomassa, predict.rf)
# 67.35409

# --- Novos casos (usando Hold out) ----

new_data <- read.csv("5-biomassa-novos-dados.csv")
View(new_data)

any(is.na(new_data))
# FALSE

new_target_data <- new_data[["biomassa"]]
new_predictors <- new_data[, colnames(new_data) != "biomassa"]

preproc_center_scale <- preProcess(new_predictors, method=c("center", "scale"))
normalized_new_predictors <- predict(preproc_center_scale, new_predictors)

normalized_new_data <- cbind(normalized_new_predictors, new_target_data)

names(normalized_new_data)[names(normalized_new_data) == "new_target_data"] <- "biomassa"
# Dados normalizados com média centralizada em 0

View(normalized_new_data)

predict.rf_ho_new_data <- predict(rf_ho, normalized_new_data)
predict.rf_ho_new_data
#        1        2        3 
# 568.6312  73.9615  45.1129

new_data$biomassa <- NULL
result <- cbind(new_data, predict.rf_new_data)
names(result)[names(result) == "predict.rf_new_data"] <- "biomassa"
View(result)
# Visualização do DF com os novos dados e a predição

# --- Geração do Gráfico de Resíduos com RF Hold Out e Dados de teste ---

test_residuals <- ((test$biomassa - predict.rf_ho) / test$biomassa) * 100

plot(predict.rf_ho, test_residuals, col = "blue", pch = 20, main = "Resíduos (%) - RF Hold Out (Dados teste)", xlab = "biomassa (estimado)", ylab = "Resíduo (%)", ylim=c(-100, 100))
abline(h = 0, col = "gray")
grid()