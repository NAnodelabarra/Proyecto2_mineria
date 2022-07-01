#Mineria de datos
#Proyecto 2
#sec. 02

#Juan de la Barra
#Diego Aguilar



#   ! PONER ARCHIVOS CSV EN LA MISMA CARPETA !
#   plots comentadas, imagenes en el repositorio


############################################################################
#######     Cargar Librerias
#

#install.packages("devtools")
##install.packages("tidyverse")
#install.packages("knitr")
#install.packages('corrplot')
#install.packages('NbClust')
#install.packages("dbscan")
#install.packages("fpc")
#install.packages('rpart')
#install.packages('rpart.plot')
#install.packages('caret')
#install.packages('recipes')
#install.packages('PerformanceAnalytics')
#install.packages('GGally')
#install.packages('regclass')
#install.packages('pROC')
#install.packages('rsample')
#install.packages("scales")
#install.packages("tidymodels")
#install.packages("broom", type="binary")
#install.packages("lubridate")
#install.packages("discrim")
#install.packages("naivebayes")

library(devtools)
library(tidyverse)
library(knitr)
library('corrplot')
library('NbClust')
library(dbscan) 
library(fpc)
library(rpart)
library(rpart.plot)
library(caret)
library(recipes)
library(PerformanceAnalytics)
library(GGally)
library(regclass)
library(pROC)
library(rsample)
library(tidymodels)
library(discrim)
library(lubridate)
library(naivebayes)



############################################################################
#######     Cargar Datos
#

aeval <- read.csv(paste(dirname(rstudioapi::getSourceEditorContext()$path), "/ALUMNOS-evalData.csv", sep=""))
atrain <- read.csv(paste(dirname(rstudioapi::getSourceEditorContext()$path), "/ALUMNOS-trainData.csv", sep=""))

#head(aeval)
#summary(aeval)



############################################################################
#######     Limpieza
#           aeval

aeval$origin <- as.factor(aeval$origin)
aeval$destination <- as.factor(aeval$destination)
aeval$date <- yday(aeval$date) #Var date ahora es dia del año
aeval$departure_time <- round(period_to_seconds(hms(aeval$departure_time))/60) #Var tiempo ahora es minuto del dia

sum(is.na(aeval))
sum(is.na(aeval$departure_time))    #todos los NA estan en deperture time
aeval <- na.omit(aeval)             #eliminamos las filas con NA
sum(is.na(aeval))
sum(is.null(aeval))

#head(aeval)
#summary(aeval)

#           atrain

atrain$origin <- as.factor(atrain$origin)
atrain$origin <- as.factor(atrain$destination)
atrain$date <- yday(atrain$date) #Var date ahora es dia del año
atrain$departure_time <- round(period_to_seconds(hms(atrain$departure_time))/60) #Var tiempo ahora es minuto del dia

sum(is.na(atrain))
sum(is.na(atrain$departure_time))    #todos los NA estan en deperture time
atrain <- na.omit(atrain)             #eliminamos las filas con NA
sum(is.na(atrain))
sum(is.null(atrain))

#head(atrain, 50)
#summary(atrain)



############################################################################
#######     Preprocesamiento
#Correlation Graphic

#Convierte noshow en al variable binaria solicitada en el encunciado
#donde es 1 si hay mas de 4 N.S.
#y 0 si hay 4 o menos ns
atrain$noshow <- as.integer(atrain$noshow)
atrain$noshow <- replace(atrain$noshow ,atrain$noshow < 5, 0)
atrain$noshow <- replace(atrain$noshow ,atrain$noshow > 4, 1)

#Selecionamos los datos
#Seleccionar los atributos que van a funcionar para el modelo ya presentado.
#Separamos las columnas en tres partes
ggpairs(head(subset(atrain, select=c(noshow,id,fligth_number,revenues_usd)),n=50000),title="correlogram with ggpairs()")

#Separamos los parametros por "tincada", primero evaluamos los que no parecen relevantes (Inultiles)
#ggpairs(head(subset(atrain, select=c(noshow,denied_boarding,date,distance,out_of_stock,departure_time,capacity)),n=50000),title="correlogram with ggpairs()")

#Luego los que parecen relevantes
#ggpairs(head(subset(atrain, select=c(noshow,bookings,denied_boarding,pax_midlow,pax_high,pax_midhigh,pax_low,pax_freqflyer,group_bookings,dom_cnx,int_cnx,p2p)),n=50000),title="correlogram with ggpairs()")

#En base a las corr vemos que las variables mas importantes para el noshow son:
#


############################################################################
#######     Procesamiento Arboil de Decision
#https://rpubs.com/jboscomendoza/arboles_decision_clasificacion


#entrenamiento_data <- subset(atrain, select = - c(atrain$origin,atrain$destination,atrain$date,atrain$departure_time))
#prueba_data <- subset(ev_data, select = - c())

entrenamiento_data <- atrain
data_split <- initial_split(entrenamiento_data, prop = 0.7)
training_data <- training(data_split) 
testing_data <- testing(data_split)
receta <- recipe(formula = noshow ~ ., data = training_data, method = "class")

####################
#Primer intento
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

#Confusion 1
cm1 <-confusionMatrix(table(test_data$noshow, test_data$predictednoshow))
test_data$predictedincome <- as.factor(test_data$predictednoshow)

table <- data.frame(confusionMatrix(test_data$noshow, test_data$predictednoshow)$table)

print(cm1)

####################
#Segunda iteracion
modelo_trees <-
  decision_tree(tree_depth = 10, min_n = 15) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

modelo_trees
fit_mod(modelo_trees)

#Confusion 2
cm2 <-confusionMatrix(table(test_data$noshow, test_data$predictednoshow))
test_data$predictedincome <- as.factor(test_data$predictednoshow)

table <- data.frame(confusionMatrix(test_data$noshow, test_data$predictednoshow)$table)

print(cm2)

####################
#Tercera iteracion
modelo_trees <-
  decision_tree(tree_depth = 7, min_n = 10) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

modelo_trees
fit_mod(modelo_trees)
#Confusion 3
cm3 <- confusionMatrix(table(test_data$noshow, test_data$predictednoshow))
test_data$predictedincome <- as.factor(test_data$predictednoshow)

table <- data.frame(confusionMatrix(test_data$noshow, test_data$predictednoshow)$table)

print(cm3)

####################
#Cuarta iteracion
modelo_trees <-
  decision_tree(tree_depth = 20, min_n = 10) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

modelo_trees
fit_mod(modelo_trees)

#Confusion 4
cm4 <-confusionMatrix(table(test_data$noshow, test_data$predictednoshow))
test_data$predictedincome <- as.factor(test_data$predictednoshow)

table <- data.frame(confusionMatrix(test_data$noshow, test_data$predictednoshow)$table)

print(cm4)

####################
#Quinta iteracion
modelo_trees <-
  decision_tree(tree_depth = 25, min_n = 10) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

modelo_trees
fit_mod(modelo_trees)

#Confusion 5
cm5 <- confusionMatrix(table(test_data$noshow, test_data$predictednoshow))
test_data$predictedincome <- as.factor(test_data$predictednoshow)

table <- data.frame(confusionMatrix(test_data$noshow, test_data$predictednoshow)$table)

print(cm)


############################################################################
#######     Procesamiento Naive Bayes
#

modelo_nb <-
  naive_Bayes(smoothness = .8) %>%
  set_engine("naivebayes")

fit_mod(modelo_nb)

############################################################################
#######     Procesamiento Linear Reg.
#
modelo_rl <- 
  logistic_reg() %>% 
  set_engine("glm")

fit_mod(modelo_rl)



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


pred_income <- predict(censo, newdata = test, type = "class")
pred_income %>% as.data.frame() %>% head()

