library('logger')
library(h2o)
library('data.table')

h2o.init()

ROOT_DIR <- getwd()

PROJECT_DIR <- 
  'Documents/github/r-prediction-rest-api'


MODELS_DIR <- 'models'
API_DIR <- 'api'
LOGS_DIR <- 'logs'

model_path_object <- 
  file.path(ROOT_DIR,
            PROJECT_DIR,
            MODELS_DIR,
            "StackedEnsemble_AllModels_AutoML_20200428_181354")

logging_file_path <- 
  file.path(ROOT_DIR,
            PROJECT_DIR,
            LOGS_DIR,
            "api_predictions.log")

log_appender(appender_file(logging_file_path))
log_layout(layout_glue_colors)
log_threshold(DEBUG)

log_info('Load saved model')
saved_model <- 
  h2o.loadModel(model_path_object)
log_info('Model loaded')


#* Return the prediction from Laymans Brothers Bank Model
#* @param LIMIT_BAL
#* @param SEX
#* @param EDUCATION
#* @param MARRIAGE
#* @param AGE
#* @param PAY_0
#* @param PAY_2
#* @param PAY_3
#* @param PAY_4
#* @param PAY_5
#* @param PAY_6
#* @param BILL_AMT1
#* @param BILL_AMT2
#* @param BILL_AMT3
#* @param BILL_AMT4
#* @param BILL_AMT5
#* @param BILL_AMT6
#* @param PAY_AMT1
#* @param PAY_AMT2
#* @param PAY_AMT3
#* @param PAY_AMT4
#* @param PAY_AMT5
#* @param PAY_AMT6
#* @post /prediction
function(LIMIT_BAL, SEX, EDUCATION, MARRIAGE,
         AGE, PAY_0, PAY_2, PAY_3, PAY_4, PAY_5,
         PAY_6, BILL_AMT1, BILL_AMT2, BILL_AMT3,
         BILL_AMT4, BILL_AMT5, BILL_AMT6, PAY_AMT1,
         PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6) {
  
  
  LIMIT_BAL <- as.numeric(LIMIT_BAL)
  SEX <- as.numeric(SEX)
  EDUCATION <- as.numeric(EDUCATION)
  MARRIAGE <- as.numeric(MARRIAGE)
  AGE <- as.numeric(AGE)
  PAY_0 <- as.numeric(PAY_0)
  PAY_2 <- as.numeric(PAY_2)
  PAY_3 <- as.numeric(PAY_3)
  PAY_4 <- as.numeric(PAY_4)
  PAY_5 <- as.numeric(PAY_5)
  PAY_6 <- as.numeric(PAY_6)
  BILL_AMT1 <- as.numeric(BILL_AMT1)
  BILL_AMT2 <- as.numeric(BILL_AMT2)
  BILL_AMT3 <- as.numeric(BILL_AMT3)
  BILL_AMT4 <- as.numeric(BILL_AMT4)
  BILL_AMT5 <- as.numeric(BILL_AMT5)
  BILL_AMT6 <- as.numeric(BILL_AMT6)
  PAY_AMT1 <- as.numeric(PAY_AMT1)
  PAY_AMT2 <- as.numeric(PAY_AMT2)
  PAY_AMT3 <- as.numeric(PAY_AMT3)
  PAY_AMT4 <- as.numeric(PAY_AMT4)
  PAY_AMT5 <- as.numeric(PAY_AMT5)
  PAY_AMT6 <- as.numeric(PAY_AMT6)
  
  
  log_debug('Request Values - LIMIT_BAL: {LIMIT_BAL} - 
  SEX: {SEX} - 
  EDUCATION: {EDUCATION} - 
  MARRIAGE: {MARRIAGE} - 
  AGE: {AGE} - 
  PAY_0: {PAY_0} - 
  PAY_2: {PAY_2} - 
  PAY_3: {PAY_3} - 
  PAY_4: {PAY_4} - 
  PAY_5: {PAY_5} - 
  PAY_6: {PAY_6} - 
  BILL_AMT1: {BILL_AMT1} - 
  BILL_AMT2: {BILL_AMT2} - 
  BILL_AMT3: {BILL_AMT3} - 
  BILL_AMT4: {BILL_AMT4} - 
  BILL_AMT5: {BILL_AMT5} - 
  BILL_AMT6: {BILL_AMT6} - 
  PAY_AMT1: {PAY_AMT1} - 
  PAY_AMT2: {PAY_AMT2} - 
  PAY_AMT3: {PAY_AMT3} - 
  PAY_AMT4: {PAY_AMT4} - 
  PAY_AMT5: {PAY_AMT5} - 
  PAY_AMT6: {PAY_AMT6}'
  )
  
  log_debug('Generate data.table...')
  predict_objects <- data.frame(
    LIMIT_BAL = c(LIMIT_BAL),
    SEX = c(SEX),
    EDUCATION = c(EDUCATION),
    MARRIAGE = c(MARRIAGE),
    AGE = c(AGE),
    PAY_0 = c(PAY_0),
    PAY_2 = c(PAY_2),
    PAY_3 = c(PAY_3),
    PAY_4 = c(PAY_4),
    PAY_5 = c(PAY_5),
    PAY_6 = c(PAY_6),
    BILL_AMT1 = c(BILL_AMT1),
    BILL_AMT2 = c(BILL_AMT2),
    BILL_AMT3 = c(BILL_AMT3),
    BILL_AMT4 = c(BILL_AMT4),
    BILL_AMT5 = c(BILL_AMT5),
    BILL_AMT6 = c(BILL_AMT6),
    PAY_AMT1 = c(PAY_AMT1),
    PAY_AMT2 = c(PAY_AMT2),
    PAY_AMT3 = c(PAY_AMT3),
    PAY_AMT4 = c(PAY_AMT4),
    PAY_AMT5 = c(PAY_AMT5),
    PAY_AMT6 = c(PAY_AMT6),
    stringsAsFactors = FALSE
  )
  
  log_debug('Convert to H20.ai Object...')
  predict_objects <- 
    as.h2o(predict_objects)
  
  log_debug('Make prediction...')
  prediction <- 
    h2o.predict(object = saved_model,
                newdata = predict_objects)
  
  prediction <- as.data.table(prediction)
  
  log_debug('Default: {prediction}')
  
  return(prediction)
}

function(req) {
  raw_body = req$postBody
  print(raw_body)
}
