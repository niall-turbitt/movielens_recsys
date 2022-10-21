# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # 01. Movie Recommender System - Data prep

# COMMAND ----------

import pyspark.sql.functions as f 
import pyspark.sql.types as t

# COMMAND ----------

movielens_home_dir = "dbfs:/home/niall/datasets/movielens20m/"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 1. MovieLens datasets in Databricks datasets
# MAGIC 
# MAGIC Dataset details can be found [here](https://grouplens.org/datasets/movielens/20m/) or at [kaggle](https://www.kaggle.com/grouplens/movielens-20m-dataset)

# COMMAND ----------

dbfs_dir = '/databricks-datasets/cs110x/ml-20m/data-001'

# Use the following csv files to create our datasets
ratings_filename = dbfs_dir + '/ratings.csv' 
movies_filename = dbfs_dir + '/movies.csv'

display(dbutils.fs.ls(dbfs_dir))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 2. Movies

# COMMAND ----------

# Create movies dataframe
movies_df_schema = t.StructType(
  [t.StructField('movie_id', t.IntegerType()),
   t.StructField('title', t.StringType()),
   t.StructField('genres', t.StringType())]
  )

movies_df = (spark.read.format('csv')
                         .options(header=True, inferSchema=False)
                         .schema(movies_df_schema)
                         .load(movies_filename)
                        )

movies_df.display()

# COMMAND ----------

# Add year col
movies_df = (movies_df
             .withColumn('year',
                         f.regexp_extract('title',r'\((\d+)\)',1))
            )

movies_df.display()

# COMMAND ----------

# Save movies_df as delta table
movies_df.write.save(movielens_home_dir + 'movies.delta')

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 3. Ratings

# COMMAND ----------

ratings_df_schema = t.StructType(
  [t.StructField('user_id', t.IntegerType()),
   t.StructField('movie_id', t.IntegerType()),
   t.StructField('rating', t.DoubleType()),
   t.StructField('timestamp', t.IntegerType())] # timestamp in unix
) 

ratings_df = (spark.read.format('csv')
              .options(header=True, inferSchema=True)
              .schema(ratings_df_schema)
              .load(ratings_filename)
              .withColumn('timestamp', f.from_unixtime(f.col('timestamp'),'yyyy-MM-dd HH:mm:ss'))
             )

ratings_df.display()

# COMMAND ----------

ratings_df.count()

# COMMAND ----------

# Save ratings_df as delta table
ratings_df.write.save(movielens_home_dir + 'ratings.delta')
