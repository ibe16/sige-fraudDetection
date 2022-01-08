#############################################################
#
# Tercera prueba
#
#############################################################

# Se van a continuar el análisis que se hizo en la segunda prueba, pero usando ahora variables categóricas

# Librerias
library(readr) # Para leer los datos
library (dplyr) # Para usar el operador %>%
library(caret)
library(corrplot) # Pintar la matiz de covarianza
library(funModeling) # Para la función df_status
library(FactoMineR) #PCA
library(factoextra) # Análisis PCA
library(mice) # Para imputar los datos
# library(VIM) # Para imputar valores usando KNN
library(Hmisc) # Para imputar valores usando la moda
library(pROC) # Para pintar la curva ROC
library(forcats)

# Semilla para que todo sea reproducible
set.seed(16)
# Cojemos un subconjutno de los dato de manera aleatoria
train <- read_csv("train_sample.csv")
train_clean <- train

# Transformamos las variables categóricas
train_clean %<>%
  mutate_at(vars(id_12:id_38, DeviceType, DeviceInfo), as.factor) %>%
  mutate_at(vars(isFraud, ProductCD, card1:card6, addr1, addr2,
                 P_emaildomain, R_emaildomain, M1:M9), as.factor)

# Deberíamos eliminar variables con más de un 50% de valores perdidos, pero vamos a hacer un eliminación un poco más fuerte
# y vamos a borrar aquellas que tengan más de un 40%
train_clean <- train_clean[, which(colMeans(!is.na(train_clean)) > 0.60)]
colnames(train_clean)
# Hemos reducido la mitad de las columnas

# Ahora nos quedamos con las variables categoricas
factor_variables <- select_if(train_clean, is.factor)
df_status(factor_variables)

# isFraud   
# ProductCD       
# card1       
# card2       
# card3       
# card4       
# card5       
# card6       
# addr1      
# addr2       
# P_emaildomain     
# M6     

summary(factor_variables)

# card1 y card2 se van a eliminar porque tienen demasiada diversidad en sus valores como para tratarlos
factor_variables$card1 <- NULL
factor_variables$card2 <- NULL

# En card3 vamos a conservar un nº limitado de categorías (5 probablemente)
factor_variables %>%
  count(card3, sort = TRUE)
# En card3 el valor 150 contiene casi todos los demás valores, así que se van a hacer 5 categorías, los NA se van a considerar
# una categoría también
factor_variables %<>% mutate(card3 = fct_explicit_na(card3)) %>% mutate(card3 = fct_lump(card3, n = 5))
factor_variables %>%count(card3, sort = TRUE)

# card4 los NA van a pasar a ser una categoría
factor_variables %>%count(card4, sort = TRUE)
factor_variables <- factor_variables %>% mutate(card4 = fct_explicit_na(card4))
factor_variables %>%count(card4, sort = TRUE)

# En card5 vamos a conservar también un nº limitado de categorías
factor_variables %>%count(card5, sort = TRUE)
# Aquí nos encontramos que los valores están ligeranmente más repartidos, vamos a mantener 10 categorías y a considerar también 
# los NA como una categoría
factor_variables %<>% mutate(card5 = fct_explicit_na(card5)) %>% mutate(card5 = fct_lump(card5, n = 10))
factor_variables %>%count(card5, sort = TRUE)

# En card6 vamos a eliminar los outliers charge card y debit or credit van a pasar a la clase mayoritaria debit
# Los NA se quedan como una categoría a parte
factor_variables %>%count(card6, sort = TRUE)
factor_variables %<>% mutate(card6 = fct_explicit_na(card6)) %>%
  mutate(card6 = fct_recode(card6, debit = "debit or credit", debit = "charge card"))
factor_variables %>%count(card6, sort = TRUE)

# addr1 vamos a quedarnos con la mitad de las categorías
factor_variables %>%count(addr1, sort = TRUE)
nlevels(factor_variables$addr1)
# La clase mayoritaría en este caso serían los valores perdidos, se van a conservar
factor_variables %<>% mutate(addr1 = fct_explicit_na(addr1)) %>% mutate(addr1 = fct_lump(addr1, n = 86))
factor_variables %>%count(addr1, sort = TRUE)

# addr2 vamos a quedarnos con la mitad de las categorías
factor_variables %>%count(addr2, sort = TRUE)
nlevels(factor_variables$addr2)
# Los NA también se van a considerar una clase ya que hay muchos
factor_variables %<>% mutate(addr2 = fct_explicit_na(addr2)) %>% mutate(addr2 = fct_lump(addr2, n = 18))
factor_variables %>%count(addr2, sort = TRUE)

# P_emaildomain se va a quedar un 6 categorias, los valores NA se van a conservar como una categoría
factor_variables %>%count(P_emaildomain, sort = TRUE)
factor_variables %<>% mutate(P_emaildomain = fct_explicit_na(P_emaildomain)) %>% mutate(P_emaildomain = fct_lump(P_emaildomain, n = 6))
factor_variables %>%count(P_emaildomain, sort = TRUE)

# M6 los valores NA van a pasar a ser una categoría
factor_variables %>%count(M6, sort = TRUE)
factor_variables <- factor_variables %>% mutate(M6 = fct_explicit_na(M6))
factor_variables %>%count(M6, sort = TRUE)

# Comprobamos como han quedado las variables
summary(factor_variables)


# Vamos a añadir las variables categóricas sin más tratamiento al resultado del PCA de la prueba anterior y vamos
# a entrenar los modelos

##################################################################################################################
##################################################################################################################

# Ahora nos quedamos con las variables numéricas
numeric_variables <- select_if(train_clean, is.numeric)
numeric_variables$isFraud <- as.numeric(ifelse(train_clean$isFraud == '1', 1, 0))

# Para eliminar las mismas filas
factor_variables$TransactionAmt <- numeric_variables$TransactionAmt

# Para ahorrar memoria eliminamos los datsets innecesarios por ahora
# rm(train, train_clean) ; invisible(gc())

# Identificar columnas con outliers
# Vamos a quitar los outliers siguiendo el EDA realizado en esta página
# https://nycdatascience.com/blog/student-works/ieee-cis-fraud-detection-detecting-fraud-from-customer-transactions/

# Las columnas donde exiten outliers son TransactionAmt, (dist1 y dist) -> categóricas
# Vamos a mostrar gráficas de estas columnas

# TansactionAmt antes de eliminar ouliers
boxplot(numeric_variables$TransactionAmt)$out
# Eliminamos los outliers https://www.r-bloggers.com/how-to-remove-outliers-in-r/
outliers <- boxplot(numeric_variables$TransactionAmt, plot=FALSE)$out
numeric_variables<- numeric_variables[-which(numeric_variables$TransactionAmt %in% outliers),]
# TraasntionAmt despues
boxplot(numeric_variables$TransactionAmt)$out

# Eliminamos las mismas filas en los factors
factor_variables<- factor_variables[-which(factor_variables$TransactionAmt %in% outliers),]
factor_variables$TransactionAmt <- NULL 

# Eliminamos avriables inncesarias para ahorrar memoria
rm(outliers, numeric_variables) ; invisible(gc())
##################################################################################################################
##################################################################################################################

# Cargamos el dataset con los resultados del PCA
load("dataset_PCA.RData")

# eliminamos una de las columnas isFraud
factor_variables$isFraud <- NULL

# Añadimos las nuevas variables
data <- cbind(comp, factor_variables)

# Eliminamos variables innecesarias
rm(comp, factor_variables, train, train_clean) ; invisible(gc())

# Entrenamos los modelos igual que en la prueba anterior

# Ahora vamos a proceder a entrenar los modelos, primero con el dataset pequeño para poder encontrar unos buenos 
# valores de hiperparametros y después con el grande

# Separamos un pequeño conjunto de muestras para entrenar el modelo
small <- sample_n(data, 5000)
freq(data = small, input='isFraud')

# Pasamos a factor isFraud
small <-
  small %>%
  mutate(isFraud = as.factor(ifelse(isFraud == 1, 'Yes', 'No')))


# Separamos en validación y train
trainIndex <- createDataPartition(small$isFraud, p = .75, list = FALSE)
train_small <- small[trainIndex, ]
val_small   <- small[-trainIndex, ]

# Random forest

# Valor de mtry estimado
sqrt(ncol(data))

# Entrenamos el modelo

# Explicar porque vamos a usar el resampling de los datos dentro del modelo
control <- trainControl(method='repeatedcv', 
                        number=10, 
                        repeats=3, 
                        search='grid',
                        classProbs = TRUE, 
                        summaryFunction = twoClassSummary,
                        sampling = "smote")

tunegrid <- expand.grid(.mtry = (1:15)) 

rf_fit_pca_factor <- train(isFraud ~ ., 
                    data = train_small,
                    method = 'rf',
                    metric = 'ROC',
                    trControl = control,
                    tuneGrid = tunegrid)
print(rf_fit_pca_factor)

rf_pca_factor_best_tune <- rf_fit_pca_factor$bestTune

# Guardamos los parámetros en un fichero
save(rf_pca_factor_best_tune, file="rf_pca_factor_best_tune.RData")
#25

# XGBOOST
control_xgb <- trainControl(method='repeatedcv', 
                            number=10, 
                            repeats=3, 
                            search='random',
                            classProbs = TRUE, 
                            summaryFunction = twoClassSummary,
                            sampling = "smote")

xgb_fit_pca_factor <- train(isFraud ~ ., 
                     data = train_small,
                     method = 'xgbTree',
                     metric = 'ROC',
                     trControl = control_xgb)
print(xgb_fit_pca_factor)

xgb_pca_factor_best_tune <- xgb_fit_pca_factor$bestTune
# 10

# Guardamos los parámetros en un fichero
save(xgb_pca_factor_best_tune, file="xgb_pca_factor_best_tune.RDS")

# Eliminamos variables innecesarias
rm(small, val_small, train_small) ; invisible(gc())

# Entrenamos los modelos con los parámetros que hemos buscado

# Ahora vamos a entrenar con todos los datos y los parámetros que hemos buscado
# Pasamos a factor isFraud
data <-
  data %>%
  mutate(isFraud = as.factor(ifelse(isFraud == 1, 'Yes', 'No')))

# Separamos en validación y train
trainIndex <- createDataPartition(data$isFraud, p = .75, list = FALSE)
train <- data[trainIndex, ]
val   <- data[-trainIndex, ]

# Random Forest
control <- trainControl(method='repeatedcv', 
                        number=5, 
                        repeats=2, 
                        search='grid',
                        classProbs = TRUE, 
                        summaryFunction = twoClassSummary,
                        sampling = "smote")

tunegrid <- rf_pca_factor_best_tune

rf_fit_pca_factor <- train(isFraud ~ ., 
                    data = train,
                    method = 'rf',
                    metric = 'ROC',
                    trControl = control,
                    tuneGrid = tunegrid)
print(rf_fit_pca_factor)
# 20
# Validamos
predictionValidationProb <- predict(rf_fit_pca_factor, val, type = "prob")
auc1 <- roc(val$isFraud, predictionValidationProb[[1]], levels = unique(val[["isFraud"]]))
roc_validation1 <- plot.roc(auc1, ylim=c(0,1), type = "S" , print.thres = T, main=paste('Validation AUC:', round(auc1$auc[[1]], 2)))

auc_rf_fit_pca_factor <- round(auc1$auc[[1]], 2)

# Guardamos el modelo
save(rf_fit_pca_factor, file="rf_pca_factor_model.RData")
save(auc_rf_fit_pca_factor, file="auc_result_rf_pca_factor.RData")

# XGBOOST
control_xgb <- trainControl(method='repeatedcv', 
                            number=5, 
                            repeats=2, 
                            search='grid',
                            classProbs = TRUE, 
                            summaryFunction = twoClassSummary,
                            sampling = "smote")

tunegrid_xgb <- xgb_pca_factor_best_tune

xgb_fit_pca_factor <- train(isFraud ~ ., 
                     data = train,
                     method = 'xgbTree',
                     metric = 'ROC',
                     trControl = control_xgb,
                     tuneGrid = tunegrid_xgb)
print(xgb_fit_pca_factor)

# Validamos
predictionValidationProb <- predict(xgb_fit_pca_factor, val, type = "prob")
auc1 <- roc(val$isFraud, predictionValidationProb[[1]], levels = unique(val[["isFraud"]]))
roc_validation1 <- plot.roc(auc1, ylim=c(0,1), type = "S" , print.thres = T, main=paste('Validation AUC:', round(auc1$auc[[1]], 2)))

auc_xgb_fit_pca_factor <- round(auc1$auc[[1]], 2)

# Guardamos el modelo
save(xgb_fit_pca_factor, file="xgb_pca_factor_model.RData")
save(auc_xgb_fit_pca_factor, file="auc_result_xgb_pca_factor.RData")