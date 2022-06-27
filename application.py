from array import array
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
from google.cloud import bigquery
import yfinance
import copy

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
st.markdown("# Industry Blog Trends WordCloud")
word_cloud1 = WordCloud(collocations = False, background_color = 'white').generate(t1)
plt.imshow(word_cloud1, interpolation='bilinear')
plt.axis("off")
plt.show()
st.pyplot()


import plotly.express as px
gdp_df = pd.read_csv("WEO_Data.csv").dropna()
gdp_df = gdp_df[gdp_df['Units'] == 'U.S. dollars']
gdp_df['GDP in 2020'] = gdp_df['2020'].apply(lambda x: x.replace(',','')).astype('float')
gdp_df = gdp_df.sort_values('GDP in 2020', ascending=False).head(20)
pop_df = pd.read_csv("API_SP.POP.TOTL_DS2_en_csv_v2_4218816.csv", on_bad_lines='skip')
gdp_df['Population'] = pop_df['2020'].dropna()

st.markdown("# Country Selection for Industry Analysis")
fig = px.scatter(gdp_df, x="GDP in 2020", y='Country',
                 hover_name="Country")
import plotly.express as px
df = px.data.gapminder()

fig1 = px.scatter(df.query("year==2007"), x="gdpPercap", y="lifeExp",
	         size="pop", color="continent",
                 hover_name="country", log_x=True, size_max=60)
st.plotly_chart(fig1)

st.markdown("# Top 20 GDP Countries")
st.plotly_chart(fig)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

@st.cache()
def getd():
    tickers = {
    'Verizon': ['USA', 'VZ'],
    'AT&T':['USA','T'], 
    'SoftBank': ['Japan', '9984.T'],
    'Deutche Telecom':['Germany','DTE.DE'], 
    'O2 Holdings': ['Germany','O2D.DE'], 
    'SK Telecom': ['South Korea','SKM'], 
    'KT Telecom': ['South Korea','KT'], 
    'Etisalat':['Middle East','7020.SR'], 
    'Du Telecom':['Middle East','DU.AE'], 
    # 'Saudi Telecom': ['Middle East', '7010.SR'],
    'TeleNet':['Belgium', 'TNET.BR'],
    'Orange Belgium':['Belgium', 'OBEL.BR'],
    'British Telecom': ['UK','BT-A.L'], 
    'Vodafone': ['UK','VOD.L'],
    'Orange France': ['France', 'ORA.PA'],
    'Buoygues' : ['France', 'EN.PA'],
    'Bell Canada': ['Canada', 'BCE.TO'],
    'Freedom Mobile': ['Canada', 'SJR-B.TO'],
    'Claro (America Movil)': ['Brazil', 'AMX'],
    'Oi Mobile': ['Brazil', 'OIBR4.SA'],
    'Telestre': ['Australia', 'TLS.AX'],
    'Indosat Ooredoo': ['Indonesia','ISAT.JK'],
    'KPN': ['Netherlands', 'KPN.AS'],
    'Swisscomm': ['Switzerland', 'SCMN.SW'],
    'Turk Telecom': ['Turkey', 'TTKOM.IS'],
    'Turkcell': ['Turkey', 'TCELL.IS'],
    'Advance Info': ['Thailand', 'ADVANC.BK'],
    'DTAC': ['Thailand', 'DTAC.BK'],
    # 'Tele2': ['Sweden', 'TEL2-A.ST'],
    # 'Telenor': ['Sweden', 'TEL.OL']
    }
    placeholder = st.empty()
    for i in tickers:
        placeholder.markdown('### **Processing**: Getting data for '+i)
        tickers[i].append(yfinance.Ticker(tickers[i][1]))
        tickers[i].append(tickers[i][2].info)  
        tickers[i].append(tickers[i][2].earnings)
    placeholder.empty()

    from functools import reduce # import needed for python3; builtin in python2
    from collections import defaultdict

    def groupBy(key, seq):
        return reduce(lambda grp, val: grp[key(val)].append(val) or grp, seq, defaultdict(list))

    arranged = groupBy(lambda x: tickers[x][0], tickers)
    return (tickers, arranged)


earnings_df = pd.DataFrame()
data = getd()
tickers = data[0]
arranged = data[1]
f=0
st.markdown("# Country Wise Competitor Analysis")

layout = dict(plot_bgcolor='white',
              margin=dict(t=20, l=20, r=20, b=20),
              xaxis=dict(title='Year',
                         linecolor='#d9d9d9',
                         showgrid=False,
                         mirror=True),
              yaxis=dict(title='Percentage Growth',
                         linecolor='#d9d9d9',
                         showgrid=False,
                         mirror=True))


data_element = st.radio("Select Data: ", ['Revenue', 'Earnings'])
l=[]

placeholder2 = st.empty()

for i in arranged:
    st.markdown(f"## *{i}*")

    for j in arranged[i]:
        earnings_df[j] = tickers[j][4][data_element].pct_change()
        st.image(tickers[j][3]['logo_url'], width = 200)
        l.append("https://www.macmillandictionary.com/external/slideshow/full/White_full.png")
    # st.image(l[:-1], width=120)
    specs = []
    for i in range(len(earnings_df.columns)):
        specs.append([{"type": "scatter"}])
    fig = make_subplots(
        rows=len(earnings_df.columns), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        specs=specs,
    )

    for i in range(len(earnings_df.columns)):
        fig.add_trace(
        go.Scatter(
            x=[2018,2019,2020,2021],
            y=earnings_df.iloc[:,i],
            mode="lines",
            name=earnings_df.columns[i] + f" {data_element} YoY Growth"
        ),
        row=i+1, col=1
        )
        
    fig.update_yaxes(title_text="Percentage Growth")
    fig.update_xaxes(title_text="Years",row=len(earnings_df.columns))
    # fig.update_yaxes(title_text="Percentage Growth", row=1, col=1)
    # fig.update_xaxes(title_text="Years", row=1, col=1)
    st.plotly_chart(fig)

    earnings_df=pd.DataFrame()
    l = []