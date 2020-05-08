
H2O K-Means from training until REST API
=====================================

##  Objects and Folders
The project is organized as follows:

 - `01-kmeans.r`: Main script for training and model serialization
 - `api`: Files regarding the endpoint and prediction functions
 - `data`: Raw data for training
 - `logs`: Stores the training log and API logs        
 - `models`: Main folder where the serialized models are stored


## Start the REST API from command line
There's some alternatives to run a `R` command from terminal. The most elegant way to run a R script in a _bash_ call it's the following one that [I unshamelessly took from the Stack Overflow](https://stackoverflow.com/questions/18306362/run-r-script-from-command-line).

    $ R < /Users/clesio/Documents/github/estatidados-h2o/src/01-kmeans/api/endpoint.R --no-save  
