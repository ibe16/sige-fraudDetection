#############################################################
#
# Segunda prueba
#
#############################################################

# Para comprobar las proporciones
# prop.isFraud <- table(train2$isFraud)
# round(prop.table(prop.isFraud),digits=2)

# En esta prueba se va a realizar un análisis de componentes principales para estimar la importancia de cada variables y
# como de buena predictora para la variable objetivo es
# 
# Se van a realizar varias pruebas
# PCA para un conjunto pequeño
# PCA para todo el dataset

# Después se realizarán las mismas pruebas usando las variables categóricas
# O se realizará otro análisis para usar estas variables

# Respecto a la prueba anterior se pueden hacer las siguientes mejoras
# Imputar más cantidad de datos
# Utilizar un balanceador mixto para no perder tantos casos
# Quitar outliers
# Normalizar los datos que se puedan

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

# Ahora nos quedamos con las variables numéricas
numeric_variables <- select_if(train_clean, is.numeric)
numeric_variables$isFraud <- as.numeric(ifelse(train_clean$isFraud == '1', 1, 0))

# Para ahorrar memoria eliminamos los datsets innecesarios por ahora
rm(train, train_clean) ; invisible(gc())

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

# Eliminamos avriables inncesarias para ahorrar memoria
rm(outliers) ; invisible(gc())

# Luego eliminamos valores de las columnas V utlizando la matriz de covarianza
# Nos quedamos solo con las V y la variable objetivo
v_variables <- select(numeric_variables, starts_with("V"))

# Quitamos los  near-zero variance predictors
nvz = nearZeroVar(v_variables)
v_variables <- v_variables[ ,-nvz]

# Calculamos la correlación
correlation <- cor(v_variables, use="complete.obs")
# Dibujamos la matriz 
corrplot(correlation, type="lower", method = "color",tl.cex = 0.1)
# Eliminamos las variables relacionadas entre si
hc = findCorrelation(correlation, cutoff=0.75) # Eliminamos por encima de 75%
hc = sort(hc)
data_to_drop = v_variables[,c(hc)]
colnames(data_to_drop)

# Quitamos las columnas descartadas de nuestro dataset
drops <- colnames(data_to_drop)
numeric_variables <- numeric_variables[ , !(names(numeric_variables) %in% drops)]


# Eliminamos avriables inncesarias para ahorrar memoria
rm(data_to_drop, nvz, hc, v_variables, drops, correlation) ; invisible(gc())

# Estado del dataset hasta ahora
df_status(numeric_variables)

# Vamos a realizar el PCA. Como es una técnica costosa en compuatación y tenemos un dataset demasiado grande vamos a coger
# un subconjunto de 3000 filas para realizar las pruebas
data_pca <- sample_n(numeric_variables[complete.cases(numeric_variables), ], 3000)
freq(data = data_pca, input='isFraud')

# Eliminamos la variable objetivo para hacer PCA
data_pca.isFraud <- data_pca$isFraud
data_pca$isFraud <- NULL

# Limpiamos las columnas con nzv que hayan podido surgir
nvz = nearZeroVar(data_pca)
data_pca <- data_pca[, -nvz]


# PCA
res_pca.sample <- PCA(data_pca, scale.unit = TRUE, ncp = 5, graph = TRUE)

# Análisis
# Eigen values
eig.val <- get_eigenvalue(res_pca.sample)
fviz_eig(res_pca.sample, addlabels = TRUE, ylim = c(0, 20))

# Explicar que son los eigen values
# Con las 31 primeras dimensiones se explica el 85% de la varianza del dataset

# Usando otra función para calcular el PCA
pca_sample <- prcomp(data_pca, scale = TRUE, center=TRUE, retx=TRUE)

# Vemos las 5 primeras componentes
head(pca_sample$rotation)[, 1:5]

# Número total de componentes
dim(pca_sample$rotation)

# Varianza explicada por cada componente
pca_sample$sdev^2

# Resumen
summary(pca_sample)
dim(pca_sample$x)

# Componentes
comp_sample <- as.data.frame(pca_sample$x)
comp_sample <- comp_sample[, 1:22]
comp_sample$isFraud <- data_pca.isFraud

# Borramos los objetos temporales
rm(data_pca, data_pca.isFraud, eig.val, pca_sample, res_pca.sample, nvz) ; invisible(gc())

# Una vez tenemos esto vamos a proceder a la imputación de valores

# Vamos a imputar por separado las V por un lado (ya que son variables encriptadas) y el resto del dataset por otro
# Estas variables tienen muchos valores perdidos y computacionalmente es muy costoso imputar datos usando MICE, knn o random foresr
# por lo que se va a usar la media o moda. También al ser valores encriptados es difícil asegurar que valor es el que se 
# ha perdido

# Columnas V
v_variables <- select(numeric_variables, starts_with("V"))
colnames(v_variables)
# Imputamos usando la moda
for(i in colnames(v_variables)){
  if(sum(is.na(v_variables[[i]])) != 0){
    v_variables[[i]] <- as.numeric(Hmisc::impute(v_variables[[i]], fun=median))
  }
}

df_status(v_variables)

# Resto de los datos
rest_of_data <- select(numeric_variables, -starts_with("V"))

# Para imputarlos vamos a usar mice
names(which(colSums(is.na(rest_of_data))>0))
imputed_data <- mice(rest_of_data, m=1, maxit = 5, method = 'cart', seed = 16)
rest_of_data <- complete(imputed_data)

df_status(rest_of_data)

# Juntamos de nuevo los dataset
data_pca <- cbind(rest_of_data, v_variables)
data_pca.isFraud <- data_pca$isFraud
data_pca$isFraud <- NULL

# ELiminamos columnas que hayan podido quedar con una varianza igual a cero
# Solo se pierde una columna
data_pca <- data_pca[ , which(apply(data_pca, 2, var) != 0)]

# Eliminamos variables intermedias
rm(imputed_data, numeric_variables, rest_of_data, v_variables, i) ; invisible(gc())

# Procedemos a realizar PCA sobre este conjunto de datos y a compararlo con el anterior

# Usando otra función para calcular el PCA
pca <- prcomp(data_pca, scale = TRUE, center=TRUE, retx=TRUE)

# Varianza explicada por cada componente
pca$sdev^2

# Resumen de las componentes
summary(pca)
dim(pca$x)

# Seleccionamos las componentes que nos hagan falta
comp <- as.data.frame(pca$x)
comp <- comp[, 1:34]
comp$isFraud <- data_pca.isFraud
save(comp, file="dataset_PCA.RData")
# Borramos los objetos que no hagan falta
rm(data_pca, pca, data_pca.isFraud) ; invisible(gc())


# Ahora vamos a proceder a entrenar los modelos, primero con el dataset pequeño para poder encontrar unos buenos 
# valores de hiperparametros y después con el grande

# Separamos un pequeño conjunto de muestras para entrenar el modelo
small <- sample_n(comp, 5000)
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
sqrt(ncol(comp_sample))
sqrt(ncol(comp))

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

rf_fit_pca <- train(isFraud ~ ., 
                       data = train_small,
                       method = 'rf',
                       metric = 'ROC',
                       trControl = control,
                       tuneGrid = tunegrid)
print(rf_fit_pca)

rf_pca_best_tune <- rf_fit_pca$bestTune


# XGBOOST
control_xgb <- trainControl(method='repeatedcv', 
                        number=10, 
                        repeats=3, 
                        search='random',
                        classProbs = TRUE, 
                        summaryFunction = twoClassSummary,
                        sampling = "smote")

xgb_fit_pca <- train(isFraud ~ ., 
                    data = train_small,
                    method = 'xgbTree',
                    metric = 'ROC',
                    trControl = control_xgb)
print(xgb_fit_pca)

xgb_pca_best_tune <- xgb_fit_pca$bestTune

# Guardamos los parámetros en un fichero
save(rf_pca_best_tune, file="rf_pca_best_tune.RDS")
save(xgb_pca_best_tune, file="xgb_pca_best_tune.RDS")

# Ahora vamos a entrenar con todos los datos y los parámetros que hemos buscado
# Pasamos a factor isFraud
data <-
  comp %>%
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

tunegrid <- rf_best_tune 

rf_fit_pca <- train(isFraud ~ ., 
                    data = train,
                    method = 'rf',
                    metric = 'ROC',
                    trControl = control,
                    tuneGrid = tunegrid)
print(rf_fit_pca)

# Validamos
predictionValidationProb <- predict(rf_fit_pca, val, type = "prob")
auc1 <- roc(val$isFraud, predictionValidationProb[[1]], levels = unique(val[["isFraud"]]))
roc_validation1 <- plot.roc(auc1, ylim=c(0,1), type = "S" , print.thres = T, main=paste('Validation AUC:', round(auc1$auc[[1]], 2)))

auc_rf_fit_pca <- round(auc1$auc[[1]], 2)

# Guardamos el modelo
save(rf_fit_pca, file="rf_pca_model.RDS")


# XGBOOST
control_xgb <- trainControl(method='repeatedcv', 
                            number=5, 
                            repeats=2, 
                            search='grid',
                            classProbs = TRUE, 
                            summaryFunction = twoClassSummary,
                            sampling = "smote")

tunegrid_xgb <- xgb_best_tune

xgb_fit_pca <- train(isFraud ~ ., 
                     data = train,
                     method = 'xgbTree',
                     metric = 'ROC',
                     trControl = control_xgb,
                     tuneGrid = tunegrid_xgb)
print(xgb_fit_pca)

# Validamos
predictionValidationProb <- predict(xgb_fit_pca, val, type = "prob")
auc1 <- roc(val$isFraud, predictionValidationProb[[1]], levels = unique(val[["isFraud"]]))
roc_validation1 <- plot.roc(auc1, ylim=c(0,1), type = "S" , print.thres = T, main=paste('Validation AUC:', round(auc1$auc[[1]], 2)))

auc_xgb_fit_pca <- round(auc1$auc[[1]], 2)

# Guardamos el modelo
save(xgb_fit_pca, file="xgb_pca_model.RDS")