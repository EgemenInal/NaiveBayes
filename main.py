from nltk.stem.porter import *
import os
from collections import Counter
from math import log
stemmer = PorterStemmer()
stop_enable=0;                      ''''enable stop words'''

class Bag_of_words:
    def __init__(self, cnt, Class,doc_count):
        self.cnt=cnt
        self.Class=Class
        self.word_count=sum(cnt.values())
        self.doc_count=doc_count

class naiveBayes:
    bags = {}
    total_words = Counter()
    total_docs = 0
    def __int__(self):
        self.bags={}
        self.total_words=Counter()
        self.total_docs=0

stop_words=[]
if stop_enable==1:
    with open("stopwords.txt","r") as fle:
        for word in fle:
            stop_words.append(word.strip())
ham_ugly=[]
spam_ugly=[]
spam=[]
ham=[]
spam_doc_count=0
ham_doc_count=0
M = []

for folder in ["spam/","ham/"]:
    for filename in os.listdir("train/"+folder):
        path=filename
        ##print("test/"+folder+path)
        with open("train/"+folder+path, 'rb') as f:
            f = f.read().decode('latin-1')
            f=f.split()
            if folder=="spam/":
                spam_doc_count+=1
            if folder=="ham/":
                ham_doc_count+=1
            for word in f:
                if folder=="spam/" and word not in stop_words:
                    spam_ugly.append(word)
                if folder=="ham/" and word not in stop_words:
                    if word not in stop_words:
                        ham_ugly.append(word)

for word in spam_ugly:
    word = stemmer.stem(word)
    spam.append(word)
for word in ham_ugly:
    word = stemmer.stem(word)
    ham.append(word)


#for m in M:
    #print(m.Word)

spam_cnt =Counter()
ham_cnt =Counter()
for w in spam:
    spam_cnt[w]+=1
'''print(spam_cnt)'''

for w in ham:
    ham_cnt[w]+=1


spam_bag=Bag_of_words(spam_cnt,"spam",spam_doc_count)
ham_bag=Bag_of_words(ham_cnt,"ham",ham_doc_count)

NaiveBBY=naiveBayes()
NaiveBBY.bags["spam/"]=spam_bag
NaiveBBY.bags["ham/"]=ham_bag
NaiveBBY.total_words=spam_cnt+ham_cnt

##print(sum(NaiveBBY.total_words.values()))
NaiveBBY.total_docs=spam_bag.doc_count+ham_bag.doc_count

def classify(naiveBayes,test_bag,k=1):
    best_class=""
    best_posterior=(-2)**29
    for label,bag in naiveBayes.bags.items():
        prior=log(bag.doc_count/naiveBayes.total_docs)
        denom=bag.word_count+(k*len(naiveBayes.total_words))
        likely=0
        for word,count in test_bag.cnt.items():
            if word not in stop_words:
                numerator=bag.cnt.get(word,0)+k
                likely+=count*log(numerator/denom)
        posterior=prior+likely
        if posterior>best_posterior:
            best_posterior=posterior
            best_class=label
    return best_class

test=[]
i=0
for folder in ["spam/","ham/"]:

    for filename in os.listdir("test/"+folder):
        path=filename
        ##print("test/"+folder+path)
        with open("test/"+folder+path, 'rb') as f:
            f = f.read().decode('latin-1')
            f=f.split()

            words=[]
            words_s=[]
            words.extend(f)

            cnt=Counter()
            for ws in words:
                if ws not in stop_words:
                    word = stemmer.stem(ws)
                    words_s.append(word)
            for w in words_s:
                if w not in stop_words:
                    cnt[w] += 1
            bag=Bag_of_words(cnt,folder,1)
            test.append(bag)

correct=0
false=0
for bag in test:
    label=bag.Class
    guess_label=classify(NaiveBBY,bag,1)
    if guess_label==label:
        correct+=1
    else:
        false+=1

print("Accuracy is %",(correct/(correct+false))*100)










'''classify(NaiveBBY,test_bag,1)'''