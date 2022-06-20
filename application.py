import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
from google.cloud import bigquery


bqclient = bigquery.Client()


table = bigquery.TableReference.from_string(
    "awesome-sylph-353909.TelecomAnalysisDataset.NewWordCloudTable"
)
rows = bqclient.list_rows(
    table,
    selected_fields=[
        bigquery.SchemaField("text", "STRING"),
    ],
)
dataframe = rows.to_dataframe(
    # Optionally, explicitly request to use the BigQuery Storage API. As of
    # google-cloud-bigquery version 1.26.0 and above, the BigQuery Storage
    # API is used by default.
    create_bqstorage_client=True,
)
print(dataframe.head())

string = ' '.join(dataframe.text.tolist())

from gensim.parsing.preprocessing import remove_stopwords
t1 = remove_stopwords(string)

# print(t1)
t1 = t1.lower()
remove_digits = 1
pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
import re
t1 = re.sub(pattern, '', t1)
st.set_option('deprecation.showPyplotGlobalUse', False)

word_cloud1 = WordCloud(collocations = False, background_color = 'white').generate(t1)
plt.imshow(word_cloud1, interpolation='bilinear')
plt.axis("off")
plt.show()
st.pyplot()
