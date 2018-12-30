import nltk 
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter 

#word tokenization and frequency using nltk 
text="Now, only two years after opening, thousands of cracks are splintering the dam’s machinery. Its reservoir is clogged with silt, sand and trees. And the only time engineers tried to throttle up the facility completely, it shook violently and shorted out the national electricity grid.This giant dam in the jungle, financed and built by China, was supposed to christen Ecuador’s vast ambitions, solve its energy needs and help lift the small South American country out of poverty.Instead, it has become part of a national scandal engulfing the country in corruption, perilous amounts of debt — and a future tethered to China.Nearly every top Ecuadorean official involved in the dam’s construction is either imprisoned or sentenced on bribery charges. That includes a former vice president, a former electricity minister and even the former anti-corruption official monitoring the project, who was caught on tape talking about Chinese bribes.Then there is the price tag: around $19 billion in Chinese loans, not only for this dam, known as Coca Codo Sinclair, but also for bridges, highways, irrigation, schools, health clinics and a half dozen other dams the government is scrambling to pay for."

#word tokenization 
t_word=word_tokenize(text) #tokenized word 

#frequency distribution 
fdist=FreqDist(t_word)
fdist.most_common(2)

#remove stopwords 
filter_sent=[]
for w in t_word:
    if w not in stop_words:
        filter_sent.append(w)

#stem words 
ps=PorterStemmer() 
stemmed_words=[]
for w in filter_sent:
    stemmed_words.append(ps.stem(w))

#word frequency 
counts=Counter(stemmed_words)
print(counts)

