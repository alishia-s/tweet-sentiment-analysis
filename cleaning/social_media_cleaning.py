#!/usr/bin/env python
# coding: utf-8

# # Social Media Aware Cleaning: Spell-checking, Link removals, Abbreviations, and Emojis/Emoticons.

# In[1]:


get_ipython().system('pip install spark-nlp==6.3.3 pyspark==3.3.1')
get_ipython().system('pip install emoji')
get_ipython().system('pip install emoticon_fix')


# In[2]:


import re
import emoji
from emoticon_fix import emoticon_fix
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, regexp_extract, col
from pyspark.sql.types import StringType
from functools import reduce
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *


# In[3]:


spark = sparknlp.start()


# In[4]:


training = spark.read.csv('../data/twitter_training.csv', header=True)
col_names = ["id", "topic", "sentiment", "content"]
training_df = training.toDF(*col_names)
training_df.printSchema()


# ## Emoji Cleaning

# ### Emojis

# In[5]:


def translate_emojis(text):
    if text is None:
        return None
    
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = text.replace("_", " ")
    text = text.replace(":", "")
    
    return text


# In[6]:


emoji_udf = udf(translate_emojis, StringType())
training_df = training_df.withColumn("content", emoji_udf(training_df["content"]))
training_df.select("content").show(5, truncate=False)


# ### Emoticons

# In[7]:


def translate_emoticons(text):
    if text is None:
        return None
    
    text = emoticon_fix(text)
    
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = text.replace("_", " ")
    text = text.replace(":", "")

    return text
    


# In[8]:


emoji_emoticon_udf = udf(translate_emoticons, StringType())
training_df = training_df.withColumn("translate_emoticons", emoji_emoticon_udf(training_df["content"]))
training_df.select("content", "translate_emoticons").show(5, truncate=False)


# ## Clean Up Website Links

# In[9]:


# Open web domains
with open("tlds.json", "r") as file:
    data = json.load(file)

tlds = data["tlds"]

# Regex for finding possible website links
website_regex = r"(http\w+)|(thesun\.\w+)|(\b\w+(?:" + "|".join(map(re.escape, tlds)) + r")\w+\b)"

training_df = training_df.withColumn(
    "content",
    regexp_replace("content", website_regex, "")
)

training_df.select("content").show(5, truncate=False)


# ## Split Hashtags

# In[10]:


def split_hashtags(text):
    if text is None:
        return None

    def split_tag(match):
        tag = match.group()[1:]
        tag = tag.replace("_", " ")
        tag = re.sub(r'([a-z])([A-Z])', r'\1 \2', tag)
        return tag

    return re.sub(r'#\w+', split_tag, text)


# In[11]:


hashtag_udf = udf(split_hashtags, StringType())
training_df = training_df.withColumn("hashtags_split", hashtag_udf(training_df["content"]))
training_df.select("content").show(5, truncate=False)


# ## Spell Check + Normalize

# In[12]:


document_assembler = DocumentAssembler()\
  .setInputCol("content")\
  .setOutputCol("document")

tokenizer = RecursiveTokenizer()\
  .setInputCols(["document"])\
  .setOutputCol("token")\
  .setPrefixes(["\"", "(", "[", "\n"])\
  .setSuffixes([".", ",", "?", ")","!", "‘s"])

normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized") \
    .setLowercase(False)\
    .setCleanupPatterns(["[^A-Za-z0-9]"])

spell_checker = NorvigSweetingModel.pretrained() \
    .setInputCols(["normalized"]) \
    .setOutputCol("spell_checked")


# In[13]:


finisher = Finisher() \
          .setInputCols("spell_checked") \
          .setOutputCols(["cleaned_text"]) \
          .setOutputAsArray(False) \
          .setCleanAnnotations(False) \
          .setAnnotationSplitSymbol(" ")

spell_check_pipeline = Pipeline(stages = [document_assembler,
                                tokenizer,
                                normalizer,
                                spell_checker,
                                finisher])


# In[14]:


training_df = spell_check_pipeline.fit(training_df).transform(training_df)
training_df.select("cleaned_text").show(5, truncate=False)


# ## Remove abbreviations

# In[15]:


with open("abbreviations.json", "r") as file:
    abbvs = json.load(file)

expr = reduce(
    lambda col, kv: regexp_replace(col, rf"(?i)\b{kv[0]}\b", kv[1]),
    abbvs.items(),
    col("cleaned_text")
)

training_df = training_df.withColumn("cleaned_text", expr)
training_df.select("cleaned_text").show(5, truncate=False)


# ## Export CSV

# In[16]:


cleaned_df = training_df.select(col("id"), 
                                col("topic"), 
                                col("sentiment").alias("og_sentiment"), 
                                col("cleaned_text"))


# In[17]:


cleaned_df.toPandas().to_csv("../cleaned_data/social_media_cleaned_tweets.csv", index=False)

