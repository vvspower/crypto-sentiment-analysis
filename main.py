
from re import S
import requests
from bs4 import BeautifulSoup
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, TFPegasusForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def calculate_overall(SENTIMENTS):
    positive = 0
    negative = 0
    overall = ""
    for result in SENTIMENTS:
        if result == "POSITIVE":
            positive += 1
        if result == "NEGATIVE":
            negative += 1
    if positive == negative:
        overall = "mixed",
    if positive > negative:
        overall = "mixed-positive"
    if negative > positive:
        overall = "mixed-negative"
    if negative == 0:
        overall = "postive"
    if positive == 0:
        overall = "negative"
    return overall


def calculate_largest(input_list):
    max = input_list[0]
    index = 0
    for i in range(len(input_list)):
        if input_list[i] > max:
            max = input_list[i]
            index = i
    return index


def process_articles(URLS):
    ARTICLES = []
    print("Processing Articles")
    for link in URLS:
        r = requests.get(link, allow_redirects=False)
        print(r.status_code)
        if r.status_code != 200:
            print("Website refused to serve")
            continue
        # print(link)
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = [paragraph.text for paragraph in paragraphs]
        words = ' '.join(text).split(' ')[:512]
        ARTICLE = ' '.join(words)
        ARTICLES.append(ARTICLE)
    return ARTICLES


def summarize(ARTICLES):
    model_name = "human-centered-summarization/financial-summarization-pegasus"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    print("Summarizing Articles")

    SUMMARIZED = []
    for x in range(len(ARTICLES)):

        input_ids = tokenizer(
            ARTICLES[x], return_tensors="pt",  truncation=True).input_ids
        output = model.generate(
            input_ids,
            max_length=32,
            num_beams=5,
            early_stopping=True
        )
        result = tokenizer.decode(output[0], skip_special_tokens=True)
        SUMMARIZED.append(result)
        print(f"Processed {x + 1} article(s)")

    print(f"summarized articles: {SUMMARIZED} ")
    return SUMMARIZED


def process_sentiment(SUMMARIZED):
    print("Processing Sentiment")
    relevant = {"0": "POSITIVE", "1": "NEGATIVE", "2": "NEUTRAL"}
    SENTIMENTS = []
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained(
        "ProsusAI/finbert")
    # tokenize text to be sent to model
    max_num = len(SUMMARIZED)
    for x in range(len(SUMMARIZED)):
        inputs = tokenizer(SUMMARIZED[x], padding=True,
                           truncation=True, return_tensors='pt')
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        model.config.id2label
        positive = predictions[:, 0].tolist()
        sentiment_value = [predictions[:, 0].tolist()[0], predictions[:, 1].tolist()[0],
                           predictions[:, 2].tolist()[0]]
        index = calculate_largest(sentiment_value)
        sentiment = relevant[str(index)]
        SENTIMENTS.append(sentiment)
        print(f"Processed {x + 1} article(s)")
    print("Completed Analysis")
    return SENTIMENTS

    # TODO: work on it


crypto = input("Please Input the CryptoCurrency: ")
URL = f"https://www.google.com/finance/quote/{crypto}-USD?hl=en"
print(f"Searching crypto news about {crypto}")
page = requests.get(URL)


soup = BeautifulSoup(page.content, "html.parser")
main = soup.find_all("main")
LINKS = []
for elements in main:
    div = elements.find_all("div", class_="z4rs2b")
    index = -1
    for a in div:
        link = a.find_all('a', href=True)
        for x in range(len(link)):
            LINKS.append(link[x].get("href"))
print(f"found {len(LINKS)} links")

articles = process_articles(LINKS)
s = summarize(articles)
result = process_sentiment(s)
print(f"Sentiments for each articles about {crypto} are {result}")
overall = calculate_overall(result)
print(f"Overall the sentiments about {crypto} are {overall}")
