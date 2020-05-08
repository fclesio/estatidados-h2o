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

# Hiperarametros do GBM 
gbm_params1 <- list(learn_rate = c(0.01, 0.1),
                    max_depth = c(3, 5, 9),
                    sample_rate = c(0.8, 1.0),
                    col_sample_rate = c(0.2, 0.5, 1.0))

# Treino e validacao do produto cartesiano do modelo GBM
gbm_grid1 <- h2o.grid("gbm",
                      x = x,
                      y = y,
                      grid_id = "estatidados_gbm_grid_1",
                      training_frame = layman_brothers.train,
                      validation_frame = layman_brothers.test,
                      ntrees = 100,
                      seed = seed,
                      hyper_params = gbm_params1)

# Pega os resultados do Grid Search e 
# ordena pelo AUC da base de validação
gbm_gridperf1 <- 
  h2o.getGrid(grid_id = "estatidados_gbm_grid_1",
              sort_by = "auc",
              decreasing = TRUE)

print(gbm_gridperf1)

# Busca o melhor modelo GBM e ordena pelo AUC
best_gbm1 <- 
  h2o.getModel(gbm_gridperf1@model_ids[[1]])

# Avaliacao do modelo com a melhor performance contra a base
# de teste
best_gbm_perf1 <-
  h2o.performance(model = best_gbm1,
                  newdata = layman_brothers.test)
h2o.auc(best_gbm_perf1)

# Os hiperparametros do melhor modelo
print(best_gbm1@model[["model_summary"]])


# Referencias
# [1] - https://stackoverflow.com/questions/9341635/check-for-installed-packages-before-running-install-packages
# [2] - http://docs.h2o.ai/h2o/latest-stable/h2o-docs/grid-search.html