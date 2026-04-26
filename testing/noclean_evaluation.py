#!/usr/bin/env python
# coding: utf-8

# ## **Evaluation of No Cleaning Method**

# In[1]:


import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from pyspark.sql.functions import trim, col

from sparknlp.pretrained import ResourceDownloader


# In[2]:


spark = sparknlp.start()


# ### Load Data

# In[3]:


CLEAN_DATA_PATH = '../cleaned_data/baseline_clean_data.csv'


# In[4]:


df = spark.read.option("header", True).csv(CLEAN_DATA_PATH)
df.show(5)
print("Number of rows:", df.count())


# ### Drop Nulls and Empty Values

# In[5]:


df = df.dropna(subset=["cleaned_text"])
print("Number of rows after dropping nulls:", df.count())


# In[6]:


from pyspark.sql.functions import trim

df = df.filter(trim(col("cleaned_text")) != "")
print("Number of rows after dropping empty strings:", df.count())


# ### Drop Rows with "Irrelevant" OG Sentiment
# These must be dropped as the model can only classify "positive", "neutral", or "negative". 

# In[7]:


df = df.filter(col("og_sentiment") != "Irrelevant")
print("Number of rows after dropping irrelevant rows:", df.count())


# ### Run Model

# In[8]:


document_assembler = DocumentAssembler() \
    .setInputCol('cleaned_text') \
    .setOutputCol('document')

tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

classifier = XlmRoBertaForSequenceClassification.pretrained('twitter_xlm_roberta_base_sentiment')\
  .setInputCols(["document",'token'])\
  .setOutputCol("pred_sentiment")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    classifier    
])

model = pipeline.fit(df)
result = model.transform(df)


# In[9]:


result.show()


# In[10]:


# View the actual labels from the pred_sentiment column
result.select("og_sentiment", "cleaned_text", "pred_sentiment.result").show()


# In[11]:


# Create column with the predicted labels
result = result.withColumn("pred_label",col("pred_sentiment")[0].result)
result.show()


# ### Normalize Label Columns

# In[12]:


from pyspark.sql.functions import lower

result = result.withColumn("og_label", lower(col("og_sentiment"))) \
               .withColumn("pred_label", lower(col("pred_label")))

result.show()


# ### Calculate Accuracy

# In[14]:


eval_df = result.select("og_label", "pred_label").toPandas()
eval_df


# In[16]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(eval_df["og_label"], eval_df["pred_label"])
print("Accuracy:", accuracy)


# ### Calculate F1 Scores

# In[17]:


from sklearn.metrics import f1_score

macro_f1 = f1_score(eval_df["og_label"], eval_df["pred_label"], average="macro")
print("Macro F1:", macro_f1)

weighted_f1 = f1_score(eval_df["og_label"], eval_df["pred_label"], average="weighted")
print("Weighted F1:", weighted_f1)


# ### Classification Report

# In[18]:


from sklearn.metrics import classification_report

print(classification_report(eval_df["og_label"], eval_df["pred_label"]))

