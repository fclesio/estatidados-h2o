#  Checa se o pacote esta instalado ou nao [1]
requiredPackages = c('h2o')
for(p in requiredPackages){
  if(!require(p,character.only = TRUE)) install.packages(p)
  library(p,character.only = TRUE)
}

# Semente randomica
seed <- 42

# Inicializa o H2O.ai
h2o.init(ip = "localhost",
         port = 54321,
         nthreads = -1,
         max_mem_size="20g",
         name = "estatidados-cluster")

# Pega a URL do Layman Brothers no Github
layman_brothers_url <- 
  "https://raw.githubusercontent.com/fclesio/learning-space/master/Datasets/02%20-%20Classification/default_credit_card.csv"

# Faz a carga no formato do H2O.ai que e.hex. 
# Não usar o metodo "h2o.uploadFile()" pois esse
# exige 4x o volume de dados em memoria antes de
# fazer a carga
layman_brothers.hex <- 
  h2o.importFile(path = layman_brothers_url,
                 destination_frame = "layman_brothers.hex")

# Como vamos trabalhar com classificacao binaria
# vamos deixar a nossa variavel dependente como
# categorica (factor)
layman_brothers.hex$DEFAULT <- 
  as.factor(layman_brothers.hex$DEFAULT)

layman_brothers.hex$EDUCATION <- 
  as.factor(layman_brothers.hex$EDUCATION)

layman_brothers.hex$MARRIAGE <- 
  as.factor(layman_brothers.hex$MARRIAGE)

# Informacoes sobre a base de dados
summary(layman_brothers.hex)

# Faz a divisao do conjunto de treinamento
# e teste na proporcao 90% para treino
# e 10% para teste
layman_brothers.split <-
  h2o.splitFrame(data = layman_brothers.hex,
                 ratios = 0.90,
                 seed = seed)

layman_brothers.train <- layman_brothers.split[[1]]
layman_brothers.test <- layman_brothers.split[[2]]


# Variavel dependente
y <- "DEFAULT"

# Variaveis independentes
x = c(
  "LIMIT_BAL"
  ,"EDUCATION"
  ,"MARRIAGE"
  ,"PAY_0"
  ,"PAY_2"
  ,"PAY_3"
  ,"PAY_4"
  ,"PAY_5"
  ,"PAY_6"
  ,"BILL_AMT1"
  ,"BILL_AMT2"
  ,"BILL_AMT3"
  ,"BILL_AMT4"
  ,"BILL_AMT5"
  ,"BILL_AMT6"
  ,"PAY_AMT1"
  ,"PAY_AMT2"
  ,"PAY_AMT3"
  ,"PAY_AMT4"
  ,"PAY_AMT5"
  ,"PAY_AMT6")

?h2o.deeplearning

?h2o.deeplearning

# Treino do modelo
dl_model <- 
  h2o.deeplearning(
    x = x,
    model_id = "estatidados_ffn",
    y = y,
    ignore_const_cols = TRUE,
    training_frame = layman_brothers.train,
    validation_frame = layman_brothers.test,
    hidden=c(100, 100),
    epochs = 50,
    l1 = 0.9,
    missing_values_handling = c("MeanImputation"),
    stopping_metric = c("AUC"),
    rate = 0.2,
    rate_annealing = 1e-06,
    categorical_encoding = c("AUTO"),
    balance_classes = TRUE,
    seed = seed)





# Informacoes sobre o modelo 
summary(dl_model)

# Predicoes
pred <- 
  h2o.predict(object = dl_model,
              newdata = layman_brothers.test)

# Sumario da classe 1 (DEFAULT)
summary(pred$p1)

# Matriz de confusao
h2o.confusionMatrix(dl_model)

# Importancia das variaveis
h2o.varimp_plot(dl_model)
dl_model@model$variable_importances

# Desempenho do modelo (Curva ROC)
perf <- 
  h2o.performance(dl_model,
                  layman_brothers.test)
plot(perf, type="roc")

# Grafico dos Coeficientes com as magnitudes padronizadas
h2o.std_coef_plot(dl_model)

# Informacooes dos Coeficientes com as magnitudes padronizadas
dl_model@model$standardized_coefficient_magnitudes

# Perda do modelo 
h2o.logloss(perf)

# Informação do AUC
h2o.auc(perf)


#############################################
# Salvando o modelo para colocar em producao
#############################################

#########
# Binario
#########
# Diretorio raiz
ROOT_DIR <- getwd()

# Diretorio do projeto
PROJECT_DIR <- 
  'Documents/github/estatidados-h2o/src/07-deep-learning'

artifact_path <- 
  file.path(ROOT_DIR,
            PROJECT_DIR
  )

model_path_object <- 
  h2o.saveModel(object=dl_model,
                path=artifact_path,
                force=TRUE
  )

print(model_path_object)

# Carrega o modelo na memoria (Binario)
saved_model <- h2o.loadModel(model_path_object)

# Predicao com o modelo recarregado em memoria
# com origem no arquivo binario
model_predict <- as.data.frame(
  h2o.predict(object = saved_model,
              newdata = layman_brothers.test,
  )
)

print(model_predict)

##############
# MOJO & POJO
##############
modelfile <- 
  h2o.download_mojo(dl_model,
                    path=artifact_path,
                    get_genmodel_jar=TRUE)

model_jar_path <- 
  paste(artifact_path, '/' ,modelfile, sep = "")


imported_model <- 
  h2o.import_mojo(mojo_file_path = model_jar_path)


# Predicao com o modelo recarregado em memoria
# com origem no arquivo MOJO
model_predict_imported <- as.data.frame(
  h2o.predict(object = imported_model,
              newdata = layman_brothers.test,
  )
)

print(model_predict_imported)

# Referencias
# [1] - https://stackoverflow.com/questions/9341635/check-for-installed-packages-before-running-install-packages