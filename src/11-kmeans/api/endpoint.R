library(plumber)
library('logger')

ROOT_DIR <- getwd()

PROJECT_DIR <- 
  'Documents/github/r-prediction-rest-api'

API_DIR <- 'api'
LOGS_DIR <- 'logs'

api_path_object <- 
  file.path(ROOT_DIR,
            PROJECT_DIR,
            API_DIR,
            "api.R")

logging_file_path <- 
  file.path(ROOT_DIR,
            PROJECT_DIR,
            LOGS_DIR,
            "api_predictions.log")

log_appender(appender_file(logging_file_path))
log_layout(layout_glue_colors)
log_threshold(DEBUG)

convert_empty <- function(string) {
  if (string == "") {
    "-"
  } else {
    string
  }
}


r <- plumb(api_path_object)


r$registerHooks(
  list(
    preroute = function() {
      # Start timer for log info
      tictoc::tic()
    },
    postroute = function(req, res) {
      end <- tictoc::toc(quiet = TRUE)
      log_info('REMOTE_ADDR: {convert_empty(req$REMOTE_ADDR)},  HTTP_USER_AGENT: "{convert_empty(req$HTTP_USER_AGENT)}", HTTP_HOST: {convert_empty(req$HTTP_HOST)},  REQUEST_METHOD: {convert_empty(req$REQUEST_METHOD)},  PATH_INFO: {convert_empty(req$PATH_INFO)}, request_status: {convert_empty(res$status)}, RESPONSE_TIME: {round(end$toc - end$tic, digits = getOption("digits", 5))}')
      
      
      
    }
  )
)

r

r$run(host="127.0.0.1", port=8000, swagger=TRUE)


# References
# [1] - https://rviews.rstudio.com/2019/08/13/plumber-logging/
# [2] - https://cran.r-project.org/web/packages/AzureContainers/vignettes/vig01_plumber_deploy.html
# [3] - https://www.rplumber.io/docs/hosting.html#docker
# [4] - https://www.rplumber.io/docs/routing-and-input.html#input-handling
# [5] - https://www.statworx.com/ch/blog/how-to-create-rest-apis-with-r-plumber/
# [6] - https://rviews.rstudio.com/2018/07/23/rest-apis-and-plumber/

