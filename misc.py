import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

content = np.array([])

from bs4 import BeautifulSoup
import requests

url = "https://inform.tmforum.org/tag/ai-data-insights/"

article_links = []

for j in range(1,8):
    req = requests.get(f"https://inform.tmforum.org/tag/ai-data-insights/page/{j}/")
    soup = BeautifulSoup(req.text, "html.parser")
    for i in soup.select(".article-item__title-link"):
        article_links.append(i.get("href"))

text = []
for i in article_links:
    req = requests.get(i)
    soup = BeautifulSoup(req.text, "html.parser")
    text.append(''.join(soup.select(".content")[0].strings))
    # text.append(driver.find_element_by_class_name("post_title").text)


content = np.array(text)

df = pd.DataFrame(content)
df.to_csv("data.csv")
