import requests
from bs4 import BeautifulSoup

url = 'https://www.goodreads.com/quotes/tag/{}?page={}'

emotions = ['love', 'inspiration']

complete = url.format(emotions[0], 1)

def get_quotes(complete):
	data = requests.get(complete)
	soup = BeautifulSoup(data.text)
	divs = soup.find_all('div', attrs={'class' : 'quoteText'})
	quotes = [div.text.strip().split('\n')[0][1:-1] for div in divs ]
	return quotes

#divs.text.strip().split('\n')[0][1:-1]	gets only the text without other info


quotes = get_quotes(complete)

x , y = [], []

for emotion in emotions:
	for i in range(1, 6):
		complete = url.format(emotion, i)
		quotes = get_quotes(complete)
		x.extend(quotes)
		y.extend([emotion] * len(quotes))
		print(f'Processed page {i} for {emotion}')
		
import pandas as pd

df = pd.DataFrame(list(zip(y,x)), columns=['emotions', 'quotes'])

df.to_csv('emotions.csv', index=False)

from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(max_features=500)

from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

tokenizer = RegexpTokenizer('\w+')
sw = set(stopwords.words('english'))
ps = PorterStemmer()

def getStemmedQuote(quote):
	quote = quote.lower()
	
	#tokenize
	tokens = tokenizer.tokenize(quote)
	
	#remove stopwords
	new_tokens = [token for token in tokens if token not in sw]
	
	stemmed_token = [ps.stem(token) for token in new_tokens]
	
	clean_quote = ' '.join(stemmed_token)
	
	return clean_quote
	
	
def getStemmedQuotes(quotes):
	d = []	
	for quote in quotes:
		d.append(getStemmedQuote(quote) )
	return d
	
x = getStemmedQuotes(x)

vect.fit(x)

x_mod = vect.transform(x).todense()
	
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_mod, y, test_size=0.33, randomstate=42)

from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB()

mode.fit(x_train, y_train)

model.score(x_test, y_test)

line = "nothing's gonna change my love for you"

x_vec = vect.transform([line]).todense()

model.predict(x_vec)


