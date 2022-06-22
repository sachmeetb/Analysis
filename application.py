import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
from google.cloud import bigquery
import yfinance

bqclient = bigquery.Client()

st.markdown(
    """
<style>
.reportview-container .markdown-text-container {
    font-family: monospace;
}
.sidebar .sidebar-content {
    background-image: linear-gradient(#2e7bcf,#2e7bcf);
    color: white;
}
.Widget>label {
    color: white;
    font-family: monospace;
}
[class^="st-b"]  {
    color: white;
    font-family: monospace;
}
.st-bb {
    background-color: transparent;
}
.st-at {
    background-color: #0c0080;
}
footer {
    font-family: monospace;
}
.reportview-container .main footer, .reportview-container .main footer a {
    color: #0c0080;
}
header .decoration {
    background-image: none;
}

</style>
""",
    unsafe_allow_html=True,
)

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


tickers = {
    'Verizon': ['USA', 'VZ'],
    'AT&T':['USA','T'], 
    'Deutche Telecom':['Germany','DTE.DE'], 
    'O2 Holdings': ['Germany','O2D.DE'], 
    'SK Telecom': ['South Korea','SKM'], 
    'KT Telecom': ['South Korea','KT'], 
    'Etisalat':['UAE','7020.SR'], 
    'Du Telecom':['UAE','DU.AE'], 
    'British Telecom': ['UK','BT-A.L'], 
    'Vodafone': ['UK','VOD.L']
}
f=0
placeholder = st.empty()
for i in tickers:
    placeholder.text('Getting data for '+i)
    tickers[i].append(yfinance.Ticker(tickers[i][1]))
    tickers[i].append(tickers[i][2].info)
    tickers[i].append(tickers[i][2].earnings)
placeholder.empty()



earnings_df = pd.DataFrame()

from functools import reduce # import needed for python3; builtin in python2
from collections import defaultdict

def groupBy(key, seq):
 return reduce(lambda grp, val: grp[key(val)].append(val) or grp, seq, defaultdict(list))

arranged = groupBy(lambda x: tickers[x][0], tickers)

l=[]
for i in arranged:
    st.write(i)
    for j in arranged[i]:
        earnings_df[j] = tickers[j][4].Earnings.pct_change()
        l.append(tickers[j][3]['logo_url'])
        l.append("https://htmlcolorcodes.com/assets/images/logo-dixon-and-moe.png")
    st.image(l[:-1], width=120)
    st.line_chart(earnings_df)
    earnings_df=pd.DataFrame()
    l = []