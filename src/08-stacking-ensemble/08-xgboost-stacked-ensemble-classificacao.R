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

# Informacoes sobre a base de dados
summary(layman_brothers.hex)

# Faz a divisao do conjunto de treinamento
# e teste na proporcao 90% para treino
# e 10% para teste
layman_brothers.split <-
  h2o.splitFrame(data = layman_brothers.hex,
                 ratios = 0.90,
                 seed = seed)

layman_brothers.train = layman_brothers.split[[1]]
layman_brothers.test = layman_brothers.split[[2]]


# Variavel dependente
y = "DEFAULT"

# Variaveis independentes
x = c(
  "LIMIT_BAL"
  ,"SEX"
  ,"EDUCATION"
  ,"MARRIAGE"
  ,"AGE"
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


# Numero de particoes para o Cross Validation
# Vai ser utilizado para o primeiro nivel do Stacking
nfolds <- 10

# Modelos Base
# Treino e validação cruzada com GBM
model_gbm <- 
  h2o.gbm(x = x,
          y = y,
          training_frame = layman_brothers.train,
          distribution = "bernoulli",
          max_depth = 3,
          min_rows = 2,
          learn_rate = 0.2,
          nfolds = nfolds,
          fold_assignment = "Modulo",
          keep_cross_validation_predictions = TRUE,
          seed = seed)

# Treino e validação cruzada com Random Forests
model_rf <- 
  h2o.randomForest(x = x,
                   y = y,
                   training_frame = layman_brothers.train,
                   nfolds = nfolds,
                   fold_assignment = "Modulo",
                   keep_cross_validation_predictions = TRUE,
                   seed = seed)

# Treino e validação cruzada com rede Deep Learning
model_dl <- 
  h2o.deeplearning(x = x,
                   y = y,
                   training_frame = layman_brothers.train,
                   l1 = 0.001,
                   l2 = 0.001,
                   hidden = c(200, 200, 200),
                   nfolds = nfolds,
                   fold_assignment = "Modulo",
                   keep_cross_validation_predictions = TRUE,
                   seed = seed)


# Modelos Base: XGBoost
# Treino e validação cruzada com XGBoost (Raso)
model_xgb_1 <- 
  h2o.xgboost(x = x,
              y = y,
              training_frame = layman_brothers.train,
              distribution = "bernoulli",
              ntrees = 50,
              max_depth = 3,
              min_rows = 2,
              learn_rate = 0.2,
              nfolds = nfolds,
              fold_assignment = "Modulo",
              keep_cross_validation_predictions = TRUE,
              seed = seed)


# Treino e validação cruzada com XGBoost (Profundo)
model_xgb_2 <- 
  h2o.xgboost(x = x,
              y = y,
              training_frame = layman_brothers.train,
              distribution = "bernoulli",
              ntrees = 50,
              max_depth = 8,
              min_rows = 1,
              learn_rate = 0.1,
              sample_rate = 0.7,
              col_sample_rate = 0.9,
              nfolds = nfolds,
              fold_assignment = "Modulo",
              keep_cross_validation_predictions = TRUE,
              seed = seed)



## Stacked Ensemble
# Aqui pegamos a lista de modelos base e vamos treinar
# os modelos treinados anteriormente
base_models <- 
  list(model_gbm@model_id,
       model_rf@model_id,
       model_dl@model_id,  
       model_xgb_1@model_id,
       model_xgb_2@model_id)

ensemble <- 
  h2o.stackedEnsemble(x = x,
                      y = y,
                      training_frame = layman_brothers.train,
                      base_models = base_models)

# Realizar a avaliação dos modelos
perf <- 
  h2o.performance(ensemble,
                  newdata = layman_brothers.test)


# Comparação do o modelo base com a base de teste
get_auc <- function(mm) h2o.auc(h2o.performance(h2o.getModel(mm), newdata = layman_brothers.test))

baselearner_aucs <- sapply(base_models, get_auc)

baselearner_best_auc_test <- max(baselearner_aucs)

ensemble_auc_test <- h2o.auc(perf)


# Comparação de desempenho entre os modelos base e o ensemble
print(sprintf("AUC - Melhor modelo base:  %s", baselearner_best_auc_test))
print(sprintf("AUC - Model Ensemble:  %s", ensemble_auc_test))


v#############################################
# Salvando o modelo para colocar em producao
#############################################

#########
# Binario
#########
# Diretorio raiz
ROOT_DIR <- getwd()

# Diretorio do projeto
PROJECT_DIR <- 
  'Documents/github/estatidados-h2o/src/08-stacking-ensemble'

artifact_path <- 
  file.path(ROOT_DIR,
            PROJECT_DIR
  )

model_path_object <- 
  h2o.saveModel(object=ensemble,
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
  h2o.download_mojo(ensemble,
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
# [2] - https://github.com/h2oai/h2o-tutorials/blob/master/tutorials/ensembles-stacking/stacked_ensemble_h2o_xgboost.Rmd
# [3] - http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html
# [4] - http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/xgboost.html