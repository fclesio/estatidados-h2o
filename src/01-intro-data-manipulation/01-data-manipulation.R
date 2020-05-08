#  Checa se o pacote esta instalado ou não [1]
requiredPackages = c('h2o')
for(p in requiredPackages){
  if(!require(p,character.only = TRUE)) install.packages(p)
  library(p,character.only = TRUE)
}

# Semente randomica global
seed <- 42

?h2o.init

h2o.init(nthreads = -1)

h2o.glm

demo(h2o.glm)

layman_brothers_url <-
  "https://raw.githubusercontent.com/fclesio/learning-space/master/Datasets/02%20-%20Classification/default_credit_card.csv"

layman_brothers.hex <-
  h2o.importFile(path = layman_brothers_url,
                 destination_frame = "layman_brothers.hex")

summary(layman_brothers.hex)


# Ver os quantis [2]
quantile(x = layman_brothers.hex$AGE, na.rm = TRUE)
h2o.hist(layman_brothers.hex$AGE)

# Agrupando os dados de educação
education_stats = h2o.group_by(data = layman_brothers.hex,
                               by = "EDUCATION",
                               nrow("EDUCATION"),
                               gb.control=list(na.methods="rm"))

education_stats.R = as.data.frame(education_stats)
education_stats.R


# Construindo base de Treino e Teste
layman_brothers.split = h2o.splitFrame(data = layman_brothers.hex,
                                       ratios = 0.70,
                                       seed = seed)

layman_brothers.train = layman_brothers.split[[1]]
layman_brothers.test = layman_brothers.split[[2]]



# Variavel dependente
Y = "DEFAULT"

# Variaveis independentes
X = c(
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


# Modelo GLM simples
layman_brothers.glm <- 
  h2o.glm(training_frame=layman_brothers.train,
          x=X,
          y=Y,
          family = "binomial",
          alpha = 0.5,
          seed = seed)

# Informações do modelo
summary(layman_brothers.glm)

# Realizando predições com o modelo
pred = h2o.predict(object = layman_brothers.glm,
                   newdata = layman_brothers.test)

# Informações de predição em relação a sua probabilidade (Classe p1 = TRUE)
summary(pred$p1)

# Matriz de confusão
h2o.confusionMatrix(layman_brothers.glm)

# Importancia das variaveis
h2o.varimp_plot(layman_brothers.glm)

# Curva ROC
perf <- h2o.performance(layman_brothers.glm,
                        layman_brothers.test)
plot(perf, type="roc")

# Grafico dos Coeficientes com as magnitudes padronizadas
h2o.std_coef_plot(layman_brothers.glm)

# Desempenho do modelo
perf <- 
  h2o.performance(layman_brothers.glm,
                  layman_brothers.test)
perf

# Perda do modelo 
h2o.logloss(perf)

# Informação do AUC
h2o.auc(perf)

# Referencias
# [1] - https://stackoverflow.com/questions/9341635/check-for-installed-packages-before-running-install-packages
# [2] - https://pt.wikipedia.org/wiki/Quantil