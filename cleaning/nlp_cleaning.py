#!/usr/bin/env python
# coding: utf-8

# ## **Extensive Cleaning Method**

# In[1]:


import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline


# In[2]:


spark = sparknlp.start()
print("Spark NLP version", sparknlp.version())
print("Apache Spark version:", spark.version)


# In[3]:


train_df = spark.read.csv('../data/twitter_training.csv')
train_df.show(5)


# In[4]:


# Add column names
col_names = ["id", "topic", "sentiment", "content"]
train_df = train_df.toDF(*col_names)
train_df.show(5)


# In[5]:


# Convert to document type
documentAssembler = DocumentAssembler() \
    .setInputCol("content") \
    .setOutputCol("document")


# In[6]:


# Tokenize the text
tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("tokens")


# In[7]:


# Correct spellings
spellChecker = NorvigSweetingModel.pretrained() \
    .setInputCols(["tokens"]) \
    .setOutputCol("spell_checked")


# In[8]:


# Normalize
normalizer = Normalizer() \
    .setInputCols(["spell_checked"]) \
    .setOutputCol("normalized")


# In[9]:


# Remove stopwords
stopwordsCleaner = StopWordsCleaner() \
    .setInputCols(["normalized"]) \
    .setOutputCol("cleaned")


# In[10]:


# Reduce to lemmas
lemmatizer = LemmatizerModel.pretrained() \
    .setInputCols(["cleaned"]) \
    .setOutputCol("lemmas")


# In[11]:


# To get the final cleaned content as text
finisher = Finisher() \
    .setInputCols(["lemmas"]) \
    .setOutputCols(["cleaned_text"]) \
    .setOutputAsArray(False) \
    .setCleanAnnotations(False)\
    .setAnnotationSplitSymbol(" ")  


# In[12]:


# Setup the pipeline
nlp_pipeline = Pipeline().setStages([
    documentAssembler, tokenizer, spellChecker, normalizer, stopwordsCleaner, lemmatizer, finisher
])


# In[13]:


# Run data through pipeline
model = nlp_pipeline.fit(train_df)
clean_train_df = model.transform(train_df)
clean_train_df.show()


# In[14]:


clean_train_df.select("cleaned_text").show()


# ### **Export cleaned data to CSV**

# In[15]:


export_df = clean_train_df.select(col("id"), 
                                  col("topic"), 
                                  col("sentiment").alias("og_sentiment"), 
                                  col("cleaned_text"))
export_df.show()


# In[16]:


export_df.toPandas().to_csv("../cleaned_data/nlp_data.csv", index=False)

