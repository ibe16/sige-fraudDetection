#############################################################
#
# Primera prueba
#
#############################################################

library (dplyr)
library(readr)
library(VIM)
library(funModeling)
library(caret)
library(GoodmanKruskal)
library(corrplot)
library(tidyverse)
#library(amap) # Para la blurt table
library(mice)
library(imputeMissings) # Para imputar valores usando la moda
library(VIM) # Para imputar valores usando KNN
library(corrplot) # Librería corrplot para dibujar la matrix de correlaciones
library(ranger) # Librería para el RF
library(pROC) # Para pintar la curva ROC
library(Information) #For WOE and IV # https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html

# Semilla para que todo sea reproducible
set.seed(16)
# Cojemos un subconjutno de los dato de manera aleatoria
train <- read_csv("train_sample.csv")
train_clean <- train
test <- read_csv("test_sample.csv")

# Convertimos las variables que sabemos que son categóricas
## Segun la descripción del dataset las siguientes columnas contiene información categórica
#ProductCD
#card1 - card6
#addr1, addr2
#Pemaildomain Remaildomain
#M1 - M9
#DeviceType
#DeviceInfo
#id12 - id38
train_clean %<>%
  mutate_at(vars(id_12:id_38, DeviceType, DeviceInfo), as.factor) %>%
  mutate_at(vars(isFraud, ProductCD, card1:card6, addr1, addr2,
                 P_emaildomain, R_emaildomain, M1:M9), as.factor)

# Miramos el estado del dataset
df_status(train_clean)

# Se eliminan las columnas con más del 5% de valores NaN
train_clean <- train_clean[, which(colMeans(!is.na(train_clean)) > 0.95)]
colnames(train_clean)

# Ahora vamos a tratar las columnas numéricas y categóricas por separado

# Primero vamos con las categóricas
# Se van a mirar por grupos, ya que tiene sentido que las que ofrezcan información del mismo tipo estén más relacionadas entre si
# Después de las variables que queden se mirará si están relacionadas, para que no se repita información

# Card1-Card6
# Docu usada para esto https://www.r-bloggers.com/to-eat-or-not-to-eat-thats-the-question-measuring-the-association-between-categorical-variables/
card_variables = select (train_clean, isFraud,card1:card6)
GK_card_varibles<- GKtauDataframe(card_variables)
plot(GK_card_varibles, corrColors = "blue")

# Con las columnas card1, card2 y card5 podemos predecir el resto

# Cogemos las variables card solo para hacer la imputación
card_clean = subset(card_variables, select = c(card1:card6))

# Vamos a imputar los valores que faltan usando Knn
card_clean <- kNN(card_clean, variable = c("card2", "card5"))
card_clean = subset(card_clean, select = c(card1,card2, card5))
df_status(card_clean)

train_clean <- subset(train_clean, select = -c(card1:card6))

train_clean <- cbind(train_clean, card_clean)

# Eliminamos los dataframes no necesarios
rm(card_variables, card_clean, train, test, GK_card_varibles) ; invisible(gc())

# La única columna que queda categórica es ProductCD, se le van a imputar valores usando la moda
productCD <- select (train_clean, ProductCD)
productCD <- impute(productCD, object = NULL, method = "median/mode", flag = FALSE)

train_clean <- subset(train_clean, select = -c(ProductCD))
train_clean <- cbind(train_clean, productCD)
colnames(train_clean)

# Eliminamos los objetos no necesarios
rm(productCD) ; invisible(gc())

# Vamos con las columnas numéricas del dataset
# Comprobamos la correlación que hay entre ellas para ver si podemos eliminar alguna

# Seleccionamos las variables numéricas de. dataset y la variable objetivo
numeric_variables <- select_if(train_clean, is.numeric)
df_status(numeric_variables)

# Normalizamos los datos
# Preguntar mañana si tiene sentido normalizar los datos

# Sacamos la matriz de correlación

# Quitamos los  near-zero variance predictors
nvz = nearZeroVar(numeric_variables)
numeric_variables_without_zero <- numeric_variables[ ,-nvz]

# Calculamos la correlación
correlation <- cor(numeric_variables_without_zero, use="complete.obs")
# Dibujamos la matriz 
corrplot(correlation, type="lower", method = "color",tl.cex = 0.1)
# Eliminamos las variables relacionadas entre si
hc = findCorrelation(correlation, cutoff=0.75) # Eliminamos por encima de 75%
hc = sort(hc)
data_to_drop = numeric_variables[,c(hc)]
colnames(data_to_drop)

# Quitamos las columnas descartadas de nuestro dataset
drops <- colnames(data_to_drop)
train_clean <- train_clean[ , !(names(train_clean) %in% drops)]

# Vemos la correlación de las variables con target

# Primero pasasmos todas las variables a número
data_numeric <- select_if(train_clean, is.numeric)
data_numeric$isFraud <- as.numeric(train_clean$isFraud)
# Calculamos la correlación
cor_target <- correlation_table(data_numeric, target='isFraud')

# Eliminamos todas las variables por debajo de 0.02
data_to_drop <- cor_target %>% 
  filter(abs(isFraud) < 0.02)

# Quitamos las columnas descartadas de nuestro dataset
drops <- data_to_drop$Variable
train_clean <- train_clean[ , !(names(train_clean) %in% drops)]

# Por último balanceamos los datos https://topepo.github.io/caret/subsampling-for-class-imbalances.html
# Comprobamos cuantos ejemplos hay de cada clase
freq(data = train_clean, input='isFraud')
sum(!complete.cases(train_clean))
data_dirty <- train_clean[rowSums(is.na(train_clean)) > 0,]
freq(data = data_dirty, input='isFraud')

#Vamos a elimar las filas con na
train_final <- na.omit(train_clean)

# Eliminamos los objetos innecesarios
rm(cor_target, correlation, data_dirty, data_numeric, data_to_drop) ; invisible(gc())
rm(numeric_variables, numeric_variables_without_zero, train_clean) ; invisible(gc())
rm(drops, hc, nvz) ; invisible(gc())

# Vamos a aplicar una transformación a las variables categóricas para el segundo modelo que vamos a entrenar
# https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
categorical_variables <- select_if(train_final, is.factor)
categorical_variables <-
  categorical_variables %>%
  mutate(isFraud = as.numeric(ifelse(isFraud == '1', 1, 0)))
IV <- create_infotables(data=categorical_variables, y="isFraud", parallel=FALSE)

train_final2 <- left_join(train_final, IV$Tables$card2, by="card2")
train_final2$card2 <- train_final2$WOE
train_final2[, c("WOE", "IV", "Percent", "N")] <- list(NULL)

train_final2 <- left_join(train_final2, IV$Tables$card5, by="card5")
train_final2$card5 <- train_final2$WOE
train_final2[, c("WOE", "IV", "Percent", "N")] <- list(NULL)

train_final2 <- left_join(train_final2, IV$Tables$ProductCD, by="ProductCD")
train_final2$ProductCD <- train_final2$WOE
train_final2[, c("WOE", "IV", "Percent", "N")] <- list(NULL)

# Eliminamos objetos intermedios
rm(categorical_variables, IV) ; invisible(gc())

# Nos aseguramos de que la variable objetivo es un factor
train_final <-
  train_final %>%
  mutate(isFraud = as.factor(ifelse(isFraud == 1, 'Yes', 'No')))

train_final2 <-
  train_final2 %>%
  mutate(isFraud = as.factor(ifelse(isFraud == 1, 'Yes', 'No')))

# Separamos en entrenamiento y validación 
trainIndex <- createDataPartition(train_final$isFraud, p = .75, list = FALSE)
train <- train_final[trainIndex, ]
train2 <- train_final2[trainIndex, ]
val   <- train_final[-trainIndex, ]
val2   <- train_final2[-trainIndex, ]

#Downsapling
down_train <- downSample(x = train,
                         y = train$isFraud)
down_train <- down_train[, -which(names(down_train) == "Class")]

down_train2 <- downSample(x = train2,
                         y = train2$isFraud)
down_train2 <- down_train2[, -which(names(down_train2) == "Class")]

freq(data = down_train, input='isFraud')


# Eliminamos los obejtos innecesarios
rm(trainIndex) ; invisible(gc())


# Como el dataset resultante es muy pequeño, vamos a ajustar los parámetros directamente con él

# Random Forest

# Comprobamos el valor estimado de mtry
sqrt(ncol(down_train))

# Entrenamos el modelo
control <- trainControl(method='repeatedcv', 
                        number=10, 
                        repeats=3, 
                        search='grid',
                        classProbs = TRUE, 
                        summaryFunction = twoClassSummary)

tunegrid <- expand.grid(.mtry = (1:15)) 

rf_fit <- train(isFraud ~ ., 
                    data = train,
                    method = 'rf',
                    metric = 'ROC',
                    trControl = control,
                    tuneGrid = tunegrid)
print(rf_fit)


rf_best_tune <- rf_fit$bestTune


# XGBOOST
control_xgb <- trainControl(method='repeatedcv', 
                            number=5, 
                            repeats=3, 
                            search='random',
                            classProbs = TRUE, 
                            summaryFunction = twoClassSummary)

xgb_fit <- train(isFraud ~ ., 
                     data = down_train,
                     method = 'xgbTree',
                     metric = 'ROC',
                     trControl = control_xgb)
print(xgb_fit)

xgb_best_tune <- xgb_fit$bestTune

save(xgb_best_tune, file="xgb_best_tune.RDS")
save(xgb_fit, file="xgb_model.RDS")
load("xgb_model.RData")
# Validamos
predictionValidationProb <- predict(xgb_fit, val2, type = "prob")
auc1 <- roc(val$isFraud, predictionValidationProb[[1]], levels = unique(val[["isFraud"]]))
roc_validation1 <- plot.roc(auc1, ylim=c(0,1), type = "S" , print.thres = T, main=paste('Validation AUC:', round(auc1$auc[[1]], 2)))

auc_xgb_fit <- round(auc1$auc[[1]], 2)

save(auc_xgb_fit, file="auc_result_xgb_fit.RData")