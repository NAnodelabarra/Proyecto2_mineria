


#######     Cargar Librerias
#


#install.packages("devtools")
library(devtools)

##install.packages("tidyverse")
library(tidyverse)

#install.packages("knitr")
library(knitr)

#install.packages('corrplot')
library('corrplot')

#install.packages('NbClust')
library('NbClust')

#install.packages("dbscan")
library(dbscan) 

#install.packages("fpc")
library(fpc)

#install.packages('rpart')
library(rpart)

#install.packages('rpart.plot')
library(rpart.plot)

#install.packages('caret')
library(caret)

#install.packages('recipes')
library(recipes)

#install.packages('PerformanceAnalytics')
library(PerformanceAnalytics)

#install.packages('GGally')
library(GGally)

#install.packages('regclass')
library(regclass)

#install.packages('pROC')
library(pROC)

#install.packages('rsample')
library(rsample)

#install.packages("scales")

#install.packages("tidymodels")
library(tidymodels)

#install.packages("broom", type="binary")

#install.packages("discrim")
library(discrim)

#install.packages("lubridate")
library(lubridate)


#######     Cargar Datos
#

aeval <- read.csv(paste(dirname(rstudioapi::getSourceEditorContext()$path), "/ALUMNOS-evalData.csv", sep="")) # nolint
atrain <- read.csv(paste(dirname(rstudioapi::getSourceEditorContext()$path), "/ALUMNOS-trainData.csv", sep="")) # nolint



#######     Preprocesamiento
#

head(aeval)
summary(aeval)



#######     Limpieza
#           aeval

aeval$origin <- as.factor(aeval$origin)
aeval$destination <- as.factor(aeval$destination)
aeval$date <- yday(aeval$date)
aeval$departure_time <- round(period_to_seconds(hms(aeval$departure_time))/60)

sum(is.na(aeval))
sum(is.na(aeval$departure_time))    #todos los NA estan en deperture time
aeval <- na.omit(aeval)             #eliminamos las filas con NA
sum(is.na(aeval))
sum(is.null(aeval))

head(aeval)
summary(aeval)

#           atrain

head(atrain)
summary(atrain)

atrain$origin <- as.factor(atrain$origin)
atrain$destination <- as.factor(atrain$destination)
atrain$date <- yday(atrain$date)
atrain$departure_time <- round(period_to_seconds(hms(atrain$departure_time))/60)

sum(is.na(atrain))
sum(is.na(atrain$departure_time))    #todos los NA estan en deperture time
atrain <- na.omit(atrain)             #eliminamos las filas con NA
sum(is.na(atrain))
sum(is.null(atrain))


#Selecionamos los datos
#Seleccionar los atributos que van a funcionar para el modelo ya presentado.

#entrenamiento_data <- subset(train_data, select = - c(Nano aca pon los valores que se van))
#prueba_data <- subset(ev_data, select = - c(aca tb))





#Separar data en train y "test" para poder analizar la factibilidad del modelo
data_split <- initial_split(atrain, prop = 0.7)

training_data <- training(data_split) 

testing_data <- testing(data_split)

#Decition trees.
#https://rpubs.com/jboscomendoza/arboles_decision_clasificacion

receta <- 
  recipe(noshow ~ ., data = training_data)

modelo_trees <-
  decision_tree(tree_depth = 5, min_n = 10) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")


fit_mod <- function(mod){
  
  modelo_fit <- 
    workflow() %>% 
    add_model(mod) %>% 
    add_recipe(receta) %>% 
    fit(data = training_data)
  
  model_pred <- 
    predict(modelo_fit, testing_data, type = "prob") %>% 
    bind_cols(testing_data) 
  
  return(model_pred %>% 
           roc_auc(truth = noshow, .pred_0))
}

fit_mod(modelo_trees)
#Confusion de este
cm1 <-confusionMatrix(table(test_data$noshow, test_data$predictednoshow))
test_data$predictedincome <- as.factor(test_data$predictednoshow)

table <- data.frame(confusionMatrix(test_data$noshow, test_data$predictednoshow)$table)

print(cm1)
#Segunda iteracion
modelo_trees <-
  decision_tree(tree_depth = 10, min_n = 15) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

modelo_trees
fit_mod <- function(mod){
  
  modelo_fit <- 
    workflow() %>% 
    add_model(mod) %>% 
    add_recipe(receta) %>% 
    fit(data = training_data)
  
  model_pred <- 
    predict(modelo_fit, testing_data, type = "prob") %>% 
    bind_cols(testing_data) 
  
  return(model_pred %>% 
           roc_auc(truth = noshow, .pred_0))
}

fit_mod(modelo_trees)
#Confusion 2
cm2 <-confusionMatrix(table(test_data$noshow, test_data$predictednoshow))
test_data$predictedincome <- as.factor(test_data$predictednoshow)

table <- data.frame(confusionMatrix(test_data$noshow, test_data$predictednoshow)$table)

print(cm2)
#Tercera iteracion
modelo_trees <-
  decision_tree(tree_depth = 7, min_n = 10) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

modelo_trees
fit_mod <- function(mod){
  
  modelo_fit <- 
    workflow() %>% 
    add_model(mod) %>% 
    add_recipe(receta) %>% 
    fit(data = training_data)
  
  model_pred <- 
    predict(modelo_fit, testing_data, type = "prob") %>% 
    bind_cols(testing_data) 
  
  return(model_pred %>% 
           roc_auc(truth = noshow, .pred_0))
}

fit_mod(modelo_trees)
#Confusion
cm3 <- confusionMatrix(table(test_data$noshow, test_data$predictednoshow))
test_data$predictedincome <- as.factor(test_data$predictednoshow)

table <- data.frame(confusionMatrix(test_data$noshow, test_data$predictednoshow)$table)

print(cm3)
#Cuarta iteracion
modelo_trees <-
  decision_tree(tree_depth = 20, min_n = 10) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

modelo_trees

fit_mod <- function(mod){
  
  modelo_fit <- 
    workflow() %>% 
    add_model(mod) %>% 
    add_recipe(receta) %>% 
    fit(data = training_data)
  
  model_pred <- 
    predict(modelo_fit, testing_data, type = "prob") %>% 
    bind_cols(testing_data) 
  
  return(model_pred %>% 
           roc_auc(truth = noshow, .pred_0))
}

fit_mod(modelo_trees)
#Confusion 4
cm4 <-confusionMatrix(table(test_data$noshow, test_data$predictednoshow))
test_data$predictedincome <- as.factor(test_data$predictednoshow)

table <- data.frame(confusionMatrix(test_data$noshow, test_data$predictednoshow)$table)


print(cm4)
#Quinta iteracion
modelo_trees <-
  decision_tree(tree_depth = 25, min_n = 10) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

modelo_trees

fit_mod <- function(mod){
  
  modelo_fit <- 
    workflow() %>% 
    add_model(mod) %>% 
    add_recipe(receta) %>% 
    fit(data = training_data)
  
  model_pred <- 
    predict(modelo_fit, testing_data, type = "prob") %>% 
    bind_cols(testing_data) 
  
  return(model_pred %>% 
           roc_auc(truth = noshow, .pred_0))
}

fit_mod(modelo_trees)
#cONFUSION 5
cm5 <- confusionMatrix(table(test_data$noshow, test_data$predictednoshow))
test_data$predictedincome <- as.factor(test_data$predictednoshow)

table <- data.frame(confusionMatrix(test_data$noshow, test_data$predictednoshow)$table)

print(cm)

############################################################################
#Confusion matrix




##############################################################
##############################################################
##############################################################
#Para una situacion real determinamos que arboles de regresion era perfecto por lo descrito a nivel teorico
#por lo tanto
#Separar data en train y "test" para poder analizar la factibilidad del modelo
#Train_data
#Ev_data
#Separacion mediante enunciado
#Decition trees.
receta <- 
  recipe(noshow ~ ., data = entrenamiento_data)

modelo_trees <-
  decision_tree(tree_depth = 5, min_n = 10) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")


fit_mod <- function(mod){
  
  modelo_fit <- 
    workflow() %>% 
    add_model(mod) %>% 
    add_recipe(receta) %>% 
    fit(data = entrenamiento_data)
  
  model_pred <- 
    predict(modelo_fit, prueba_data, type = "prob") %>% 
    bind_cols(prueba_data) 
  
  return(model_pred %>% 
           roc_auc(truth = noshow, .pred_0))
}

fit_mod(modelo_trees)
##############################################################################
##############################################################################
##############################################################################