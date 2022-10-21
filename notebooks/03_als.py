# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # 03. Movie Recommender System - Train and tune ALS

# COMMAND ----------

import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient

import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

movielens_home_dir = "dbfs:/home/niall/datasets/movielens20m/"

ratings_df = spark.read.load(movielens_home_dir + 'ratings.delta').sample(0.5)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 01. Single run training

# COMMAND ----------

def train_als(ratings_df, 
              split_prop, 
              max_iter, 
              reg_param, 
              rank,
              cold_start_strategy="drop",
              seed=42):

    mlflow.autolog()
    
    (training_df, test_df) = ratings_df.randomSplit([split_prop, 1 - split_prop], seed=seed)
    
    # Cache both training and test datasets
    training_df.cache()
    test_df.cache()
    
    with mlflow.start_run(run_name='als') as run:

        train_count = training_df.count()
        mlflow.log_metric("train_count", train_count)
        test_count = test_df.count()
        mlflow.log_metric("test_count", test_count)

        print(f"train_count: {train_count}, test_count: {test_count}")

        als = (
          ALS()
          .setUserCol("user_id")
          .setItemCol("movie_id")
          .setRatingCol("rating")
          .setPredictionCol("predictions")
          .setMaxIter(max_iter)
          .setSeed(seed)
          .setRegParam(reg_param)
          .setRank(rank)
          .setColdStartStrategy(cold_start_strategy)
        )

        print("Training ALS model...")
        als_model = Pipeline(stages=[als]).fit(training_df)
        mlflow.spark.log_model(als_model, "als_model")
        print("Training finished!")

        reg_eval = RegressionEvaluator(predictionCol="predictions", labelCol="rating", metricName="mse")
        print("Computing predictions against test set...")
        pred_test_df = als_model.transform(test_df)

        print('Evaluating predictions...')
        train_mse = reg_eval.evaluate(als_model.transform(training_df))
        mlflow.log_metric("train_mse", train_mse)
        print(f"train MSE: {train_mse}")        
        test_mse = reg_eval.evaluate(pred_test_df)
        mlflow.log_metric("test_mse", test_mse)
        print(f"test MSE: {test_mse}")
        
    return pred_test_df

# COMMAND ----------

pred_test_df = train_als(ratings_df, 
                         split_prop=0.8, 
                         max_iter=5, 
                         reg_param=0.1, 
                         rank=10, 
                         seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 02. Tuning with Hyperopt
# MAGIC 
# MAGIC 
# MAGIC In the following section we create the Hyperopt workflow to tune our distributed DL training process. We will:
# MAGIC * Define an objective function to minimize
# MAGIC * Define a search space over hyperparameters
# MAGIC * Specify the search algorithm and use `fmin()` to tune the model
# MAGIC 
# MAGIC For more information about the Hyperopt APIs, see the [Hyperopt docs](https://github.com/hyperopt/hyperopt/wiki/FMin).

# COMMAND ----------

from hyperopt import hp
from hyperopt import STATUS_OK
from hyperopt import fmin, tpe, STATUS_OK

# COMMAND ----------

split_prop = 0.8
seed = 42

(training_df, test_df) = ratings_df.sample(0.1).randomSplit([split_prop, 1 - split_prop], seed=seed)

# Cache both training and test datasets
training_df.cache()
test_df.cache()

train_count = training_df.count()
test_count = test_df.count()

print(train_count, test_count)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define objective function
# MAGIC 
# MAGIC First, we need to [create an **objective function**](http://hyperopt.github.io/hyperopt/getting-started/minimizing_functions/). This is the function that Hyperopt will call for each set of inputs.
# MAGIC 
# MAGIC The basic requirements are:
# MAGIC 
# MAGIC 1. An **input** `params` including hyperparameter values to use when training the model
# MAGIC 2. An **output** containing a loss metric on which to optimize

# COMMAND ----------

def tune_als(max_iter: float, reg_param: float, rank: float):
    
    seed=42
    mlflow.autolog()
    run_name = f"max_iter_{max_iter}_reg_param_{reg_param}_rank_{rank}"   
        
    with mlflow.start_run(run_name=run_name, nested=True) as run:        

        mlflow.log_metric("train_count", train_count)
        mlflow.log_metric("test_count", test_count)

        print(f"train_count: {train_count}, test_count: {test_count}")

        als = (
          ALS()
          .setUserCol("user_id")
          .setItemCol("movie_id")
          .setRatingCol("rating")
          .setPredictionCol("predictions")
          .setMaxIter(max_iter)
          .setSeed(seed)
          .setRegParam(reg_param)
          .setRank(rank)
          .setColdStartStrategy("drop")
        )

        print("Training ALS model...")
        print(f"max_iter: {max_iter}") 
        print(f"reg_param: {reg_param}")
        print(f"rank: {rank}")

        als_model = Pipeline(stages=[als]).fit(training_df)
        mlflow.spark.log_model(als_model, "als_model")
        print("Training finished!")

        reg_eval = RegressionEvaluator(predictionCol="predictions", labelCol="rating", metricName="mse")
        print("Computing predictions against test set...")
        pred_test_df = als_model.transform(test_df)

        print('Evaluating predictions...')
        train_mse = reg_eval.evaluate(als_model.transform(training_df))
        mlflow.log_metric("train_mse", train_mse)
        print(f"train MSE: {train_mse}")        
        test_mse = reg_eval.evaluate(pred_test_df)
        mlflow.log_metric("test_mse", test_mse)
        print(f"test MSE: {test_mse}")

    return test_mse

# COMMAND ----------

def objective_function(params: dict):
    """
    This method is passed to hyperopt.fmin().

    :param params: hyperparameters. Its structure is consistent with how search space is defined. See below.
    :return: dict with fields 'loss' (scalar loss) and 'status' (success/failure status of run)
    """    
    test_mse = tune_als(**params)
    
    return {'loss': test_mse, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC If you have a metric that you want to *maximize*, you will have to negate the loss value returned. E.g if `test_mse` here was instead `roc_auc`, then the loss value would have to be `-roc_auc`

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the search space
# MAGIC 
# MAGIC Next, we need to [define the **search space**](http://hyperopt.github.io/hyperopt/getting-started/search_spaces/).
# MAGIC 
# MAGIC This example tunes 3 hyperparameters: `max_iter`, `rank` and `reg_param`. See the [Hyperopt docs](https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions) for details on defining a search space and parameter expressions.

# COMMAND ----------

search_space = {
    'reg_param': hp.loguniform('reg_param', -5, 0),
    'max_iter': hp.quniform('max_iter', 5, 15, 1),
    'rank': hp.quniform('rank', 5, 15, 1)
}

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Tune the model using Hyperopt `fmin()`
# MAGIC 
# MAGIC - Set `max_evals` to the maximum number of points in hyperparameter space to test, that is, the maximum number of models to fit and evaluate. Because this command evaluates many models, it will take several minutes to execute.
# MAGIC - You must also specify which search algorithm to use. The two main choices are:
# MAGIC   - `hyperopt.tpe.suggest`: Tree of Parzen Estimators, a Bayesian approach which iteratively and adaptively selects new hyperparameter settings to explore based on previous results
# MAGIC   - `hyperopt.rand.suggest`: Random search, a non-adaptive approach that randomly samples the search space

# COMMAND ----------

# MAGIC %md 
# MAGIC **Important:** We are training trials (each distributed training process) in a *sequential* manner. As such, when using Hyperopt with Spark ML, do not pass a `trials` argument to `fmin()`. 
# MAGIC 
# MAGIC When you do not include the `trials` argument, Hyperopt uses the default `Trials` class, which runs on the cluster driver. Hyperopt needs to evaluate each trial on the driver node so that each trial can initiate distributed training jobs.  The `SparkTrials` class is incompatible with distributed training jobs, as it evaluates each trial on one worker node.

# COMMAND ----------

with mlflow.start_run(run_name='hyperopt_tuning_als') as hyperopt_mlflow_run:
        
    # The number of models we want to evaluate
    max_evals = 4

    # Run the optimization process
    best_hyperparam = fmin(
        fn=objective_function, 
        space=search_space,
        algo=tpe.suggest, 
        max_evals=max_evals,
    )

    # Log optimal hyperparameter values
    mlflow.log_param('reg_param', best_hyperparam['reg_param'])
    mlflow.log_param('max_iter', best_hyperparam['max_iter'])
    mlflow.log_param('rank', best_hyperparam['rank'])    

# COMMAND ----------

# Print out the parameters that produced the best model
print(best_hyperparam)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## iv. Get the best run via MLflow Tracking

# COMMAND ----------

# Get the run_id of the parent run under which the Hyperopt trials were tracked
parent_run_id = hyperopt_mlflow_run.info.run_id

# Return all trials (tracked as child runs) as a pandas DataFrame, and order by accuracy descending
hyperopt_trials_pdf = (mlflow.search_runs(filter_string=f'tags.mlflow.parentRunId="{parent_run_id}"', 
                                          order_by=['metrics.test_mse']))

# The best trial will be the first row of this pandas DataFrame
best_run = hyperopt_trials_pdf.iloc[0]
best_run_id = best_run['run_id']

print(f'MLflow run_id of best trial: {best_run_id}')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## v. Register best run to MLfow Model Registry

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We register our model to the MLflow Model Registry to use in our subsequent inference notebook.

# COMMAND ----------

# Define unique name for model in the Model Registry
registry_model_name = 'niall_movielens'

# Register the best run. Note that when we initially register it, the model will be in stage=None
model_version = mlflow.register_model(f'runs:/{best_run_id}/als_model', 
                                      name=registry_model_name)

# COMMAND ----------

# Transition model to stage='Production'
client = MlflowClient()
client.transition_model_version_stage(
    name=registry_model_name,
    version=model_version.version,
    stage='Production'
)

# COMMAND ----------

# Load model from the production stage in MLflow Model Registry
prod_als_model = mlflow.spark.load_model(f'models:/{registry_model_name}/production')

# COMMAND ----------

pred_test_df = prod_als_model.transform(test_df)

pred_test_df.display()
