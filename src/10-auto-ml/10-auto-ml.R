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

# Treino do modelo
aml <- h2o.automl(x=X,
                  y=Y,
                  training_frame = layman_brothers.train,
                  validation_frame = layman_brothers.test,
                  max_models = 30,
                  # leaderboard_frame = layman_brothers.test,
                  # blending_frame = NULL,
                  nfolds = 5,
                  # fold_column = NULL,
                  # weights_column = NULL,
                  # balance_classes = TRUE,
                  # class_sampling_factors = sample_factors,
                  # max_after_balance_size = 5,
                  # max_runtime_secs = 36000,
                  # max_runtime_secs_per_model = 600,
                  # max_models = NULL,
                  stopping_metric = c("AUC"),
                  # stopping_tolerance = NULL,
                  # stopping_rounds = 3,
                  project_name = "estatidados-auto-ml",
                  exclude_algos = c("DeepLearning"),
                  # include_algos = NULL,
                  # modeling_plan = NULL,
                  # exploitation_ratio = 0.1,
                  # monotone_constraints = NULL,
                  # algo_parameters = NULL,
                  # keep_cross_validation_predictions = FALSE,
                  # keep_cross_validation_models = FALSE,
                  # keep_cross_validation_fold_assignment = FALSE,
                  sort_metric = c("AUC"),
                  # export_checkpoints_dir = NULL,
                  verbosity = "warn",
                  seed = seed
                  )


# AutoML Leaderboard
lb <- aml@leaderboard

# Informacoes do Leaderboard
print(lb, n = nrow(lb))

# Melhor modelo
aml@leader

# Predicao como melhor modelo
pred <- h2o.predict(aml@leader, layman_brothers.test)

# AUC do melhor modelo
auc.basic_valid <- h2o.auc(aml@leader, train=TRUE, valid=TRUE, xval=FALSE)
auc.basic_valid

# Referencias
# [1] - https://stackoverflow.com/questions/9341635/check-for-installed-packages-before-running-install-packages
# [2] - http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
# [3] - http://docs.h2o.ai/h2o/latest-stable/h2o-docs/parameters.html


