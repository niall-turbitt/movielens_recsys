# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # 02. Movie Recommender System - Data exploration

# COMMAND ----------

movielens_home_dir = "dbfs:/home/niall/datasets/movielens20m/"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 01. Movies 

# COMMAND ----------

movies_df = spark.read.load(movielens_home_dir + 'movies.delta')
movies_df.createTempView('movies')

# COMMAND ----------

# MAGIC %sql select count(*) from movies

# COMMAND ----------

# MAGIC %sql select movie_id, title, year, genres from movies

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Let's see how many movies there are each year in this set.

# COMMAND ----------

# MAGIC %sql select year, count(*) from movies group by year# order by year asc

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 02. Ratings 

# COMMAND ----------

ratings_df = spark.read.load(movielens_home_dir + 'ratings.delta')
ratings_df.createTempView('ratings')

# COMMAND ----------

# MAGIC %sql select * from ratings

# COMMAND ----------

# MAGIC %sql select count(*) from ratings

# COMMAND ----------

# MAGIC %sql select count(*) from ratings

# COMMAND ----------

# MAGIC %sql select count(distinct(user_id)), count(distinct(movie_id)) from ratings

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Let's figure out which movies are most commonly rated - we'll need to join on the movies table to get the name and year.

# COMMAND ----------

# MAGIC %sql 
# MAGIC select 
# MAGIC   ratings.movie_id, movies.title, movies.year, count(*) as ratings_count 
# MAGIC from 
# MAGIC   ratings
# MAGIC join
# MAGIC   movies
# MAGIC group by 
# MAGIC   ratings.movie_id, movies.title, movies.year
# MAGIC order by 
# MAGIC   ratings_count desc
