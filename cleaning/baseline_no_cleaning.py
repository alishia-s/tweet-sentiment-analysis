#!/usr/bin/env python
# coding: utf-8

# In[15]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os


# In[16]:


spark = SparkSession.builder.getOrCreate()


# In[17]:


train_df = spark.read.csv('../data/twitter_training.csv')


# In[18]:


col_names = ["id", "topic", "sentiment", "content"]
train_df = train_df.toDF(*col_names)
train_df.show(5, truncate=False)


# In[19]:


baseline_df = train_df.withColumn("baseline_content", col("content"))


# In[20]:


baseline_df.select("content", "baseline_content").show(5, truncate=False)


# In[21]:


baseline_train_df = baseline_df.select("id", "topic", "sentiment", "content", "baseline_content")
baseline_train_df.show(5, truncate=False)


# ### **Export cleaned data to CSV**

# In[ ]:


os.makedirs("../cleaned_data", exist_ok=True)

export_df = baseline_train_df.select(
    col("id"),
    col("topic"),
    col("sentiment").alias("og_sentiment"),
    col("baseline_content").alias("cleaned_text") 
)

export_df.toPandas().to_csv("../cleaned_data/baseline_data.csv", index=False)

