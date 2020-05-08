if (!require('logger')) install.packages('logger'); library('logger')

# Local directories 
ROOT_DIR <- getwd()

PROJECT_DIR <- 
  'Documents/github/estatidados-h2o/src/01-kmeans'

DATA_DIR <- 'data'
MODELS_DIR <- 'models'
API_DIR <- 'api'
LOGS_DIR <- 'logs'

get_artifact_path <- function(file_name,
                              artifact_dir,
                              root_dir=ROOT_DIR,
                              project_dir=PROJECT_DIR){
  artifact_path <- 
    file.path(root_dir,
              project_dir,
              artifact_dir,
              file_name)
  
  return (artifact_path)
}


logging_file_path <- 
  get_artifact_path("training_pipeline_kmeans.log", LOGS_DIR)

log_appender(appender_file(logging_file_path))
log_layout(layout_glue_colors)
log_threshold(DEBUG)

log_info('Start logging...')

r_version <- R.Version()$version.string
log_debug('R Version: {r_version}')

session_info_os <- sessionInfo()$running
log_debug('Session Info OS: {session_info_os}')


log_info('Instantiate function to install dependencies...')
install_dependencies <- function(){
  
  package_url_logger <- 'https://cran.rstudio.com/bin/macosx/el-capitan/contrib/3.6/logger_0.1.tgz'
  package_url_h2o <- 'https://cran.rstudio.com/bin/macosx/el-capitan/contrib/3.6/h2o_3.30.0.1.tgz'
  package_url_cluster <- 'https://cran.rstudio.com/bin/macosx/el-capitan/contrib/3.6/cluster_2.1.0.tgz'
  package_url_dplyr <- 'https://cran.rstudio.com/bin/macosx/el-capitan/contrib/3.6/dplyr_0.8.5.tgz'
  package_url_tidyverse <- 'https://cran.rstudio.com/bin/macosx/el-capitan/contrib/3.6/tidyverse_1.3.0.tgz'
  
  log_debug('logger CRAN URL: {package_url_logger}')
  log_debug('h2o CRAN URL: {package_url_h2o}')
  log_debug('cluster CRAN URL: {package_url_cluster}')
  log_debug('dplyr CRAN URL: {package_url_dplyr}')
  log_debug('tidyverse CRAN URL: {package_url_tidyverse}')
  
  packages_urls <- c(
    package_url_logger,
    package_url_dplyr,
    package_url_cluster,
    package_url_tidyverse,
    package_url_h2o
  )
  
  for(url in packages_urls)
  {for(package_url in url)
    log_info('Installing {} package')
    {install.packages(package_url, repos=NULL, type='source')}
    log_info('Package {package_url} installation finished')
    }
}

log_info('Start installing dependencies...')
install_dependencies()
log_info('Dependencies installed...')

packageVersion_logger <- packageVersion('logger')[1]
packageVersion_h2o <- packageVersion('h2o')[1]
packageVersion_cluster <- packageVersion('cluster')[1]
packageVersion_dplyr <- packageVersion('dplyr')[1]
packageVersion_tidyverse <- packageVersion('tidyverse')[1]

log_debug('logger Version: {packageVersion_logger}')
log_debug('h2o Version: {packageVersion_h2o}')
log_debug('cluster Version: {packageVersion_cluster}')
log_debug('dplyr Version: {packageVersion_dplyr}')
log_debug('tidyverse Version: {packageVersion_tidyverse}')


log_info('Loading packages...')
packages <- c(
  "logger",
  "h2o",
  "cluster",
  "dplyr",
  "tidyverse")
invisible(lapply(packages, library, character.only = TRUE))
log_info('Packages loaded')


session_info_base_packages <- sessionInfo()$basePkgs
log_info('Session Info Base Packages: {session_info_base_packages}')

session_info_loaded_packages <- sessionInfo()$loadedOnly
log_debug('Session Info Loaded Packages: {session_info_loaded_packages}')


log_info('Initializing H2O...')
host = "localhost"
host_port = 54321
cpus = 11
memory_size = "7g"

log_debug('H2O Cluster host: {host}')
log_debug('H2O Cluster host port: {host_port}')
log_debug('H2O Cluster Number CPUs: {cpus}')
log_debug('H2O Cluster Memory Size allocated: {memory_size}')
  
h2o.init(
  ip = host,
  port = host_port,
  nthreads = cpus,
  max_mem_size = memory_size
)

cluster_status <- h2o.clusterStatus()
log_debug('H2O Cluster Status Info: {cluster_status}')

log_info("Fetching data...")
training_data_path <- 
  get_artifact_path("Phones_accelerometer.csv", DATA_DIR)
log_debug("Data path {training_data_path}")

phones_accelerometer.hex <- 
  h2o.uploadFile(path = training_data_path)
log_info("Data Fetched...")

log_info("Creating feature...")
phones_accelerometer.hex$time_diff <- 
  phones_accelerometer.hex$Arrival_Time - 
  phones_accelerometer.hex$Creation_Time
log_info("Feature created...")


log_info("Starting Training...")
phones_accelerometer.km <- 
  h2o.kmeans(training_frame = phones_accelerometer.hex,
             k = 20,
             x = c("x", "y", "z", "User", "Model", "Device", "gt", "time_diff"),
             model_id='kmeans_estatidados',
             categorical_encoding = "AUTO",
             estimate_k=TRUE,
             max_iterations=1000,
             seed=42,
             standardize = TRUE,
             score_each_iteration = TRUE
  )
log_info("Training finished")

# Model details
print(phones_accelerometer.km)

model_parameters <- phones_accelerometer.km@parameters
log_info("Model Parameters: {model_parameters}")

log_info("Pass the data to the predict function...")
kmeans_predict <- 
  h2o.predict(object = phones_accelerometer.km,
              newdata = phones_accelerometer.hex,
  )


log_info("Assign predictions in the original DF...")
phones_accelerometer.hex$CLUSTER <-
  kmeans_predict


log_info("Save file in CSV...")
phones_accelerometer_clusters_path <- 
  get_artifact_path("phones_accelerometer_clusters.csv", DATA_DIR)

h2o.exportFile(phones_accelerometer.hex,
               phones_accelerometer_clusters_path,
               force = FALSE,
               parts = 1)
log_info("File saved in: {phones_accelerometer_clusters_path}")

#####################################
# Save model for Productionizing [4]
#####################################

#########
# Binary
#########
model_path <- 
  get_artifact_path("", MODELS_DIR)

model_path_object <- 
  h2o.saveModel(object=phones_accelerometer.km,
                path=model_path,
                force=TRUE
  )

print(model_path_object)

# Load the model (Binary)
saved_model <- h2o.loadModel(model_path_object)

# Predict with the loaded model from binary
kmeans_predict <- as.data.frame(
  h2o.predict(object = saved_model,
              newdata = phones_accelerometer.hex,
  )
)

print(kmeans_predict)

##############
# MOJO & POJO
##############
modelfile <- 
  h2o.download_mojo(phones_accelerometer.km,
                    path=model_path,
                    get_genmodel_jar=TRUE)

model_jar_path <- 
  paste(model_path, modelfile, sep = "")

# We will get the error [4] 
# No donuts for the K-Means
imported_model <- 
  h2o.import_mojo(mojo_file_path = model_jar_path)

# Predict with the loaded model from binary
# kmeans_predict <- as.data.frame(
#   h2o.predict(object = imported_model,
#               newdata = phones_accelerometer.hex,
#   )
# )

# References
# [0] - Dataset: https://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition
# [1] - https://towardsdatascience.com/an-efficient-way-to-install-and-load-r-packages-bc53247f058d
# [2] - https://stackoverflow.com/questions/15956183/how-to-save-a-data-frame-as-csv-to-a-user-selected-location-using-tcltk
# [3] - https://stackoverflow.com/questions/13110076/function-to-concatenate-paths
# [4] - http://docs.h2o.ai/h2o/latest-stable/h2o-docs/productionizing.html
# [5] - http://manishbarnwal.com/blog/2016/10/05/install_a_package_particular_version_in_R/
# [6] - https://stackoverflow.com/questions/4090169/elegant-way-to-check-for-missing-packages-and-install-them
# [7] - https://rstudio.github.io/packrat/
# [8] - https://daroczig.github.io/logger/
# [9] - https://stackoverflow.com/questions/2031163/when-to-use-the-different-log-levels
# [10] - https://stackoverflow.com/questions/12193779/how-to-write-trycatch-in-r
# [11] - http://adv-r.had.co.nz/Exceptions-Debugging.html

# Clean up all objects
h2o.removeAll()
