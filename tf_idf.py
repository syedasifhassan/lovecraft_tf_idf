import os
import re
import numpy
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

mythos_refs = {"cthulhu":[],
               "nyarlathotep":[],
               "azathoth":[],
               "shub-niggurath":[],
               "necronomicon":[],
               "hazred":[],
               "yig":[],
               "hastur":[],
#               "ithaqua":[],
               "yog-sothoth":[],
               "sothoth":[],
#               "shudde m'ell":[],
#               "abhoth":[],
#               "glaaki":[],
               "tsathoggua":[]
               }
mythos_res = {}
for mythos_ref in mythos_refs.keys():
    mythos_re = re.compile(mythos_ref,re.IGNORECASE or re.MULTILINE)
    mythos_res[mythos_ref] = mythos_re
            

print ("files available:")
mypath = "../lovecraftcorpus-master/"
(_, _, filenames) = next(os.walk(mypath))
print(filenames)

print("reading files")
sentence_corpus = []
paragraph_corpus = []
story_titles = []
story_corpus = []
story_lump = [""]
setup_corpus = []
setup_lump = [""]
climax_corpus = []
climax_lump = [""]
for filename in filenames:
    full_filename = mypath+filename
    f=open(full_filename, "r")
    if f.mode =='r':
        contents = f.read()
        contents = re.sub("\n+","\n",contents)
        story_lump[0] += " "
        story_lump[0] += contents
        paragraphs = contents.split('\n')
        paragraph_corpus += paragraphs
        (title,maintext) = contents.split('\n',1)
        titlel = title.lower()
        print(titlel)
        story_titles += [titlel]
        story_corpus += [contents]
        for mythos_ref in mythos_refs.keys():
            if mythos_res[mythos_ref].search(contents):
                mythos_refs[mythos_ref] += [titlel]
        sentence_corpus += [titlel]
        sentences = maintext.split('.')
        sentence_corpus += sentences
        N_sentences = len(sentences)
        #print(int(len(sentences)*2/3))
        climax_index = int(len(sentences)*2/3)
        setup_sentences = sentences[:climax_index-1]
        setup_contents = titlel + ' '.join(setup_sentences)
        setup_corpus += [setup_contents]
        setup_lump[0] += " "
        setup_lump[0] += setup_contents
        climax_sentences = sentences[climax_index:]
        climax_contents = titlel + ' '.join(climax_sentences)
        climax_corpus += [climax_contents]
        climax_lump[0] += " "
        climax_lump[0] += climax_contents
#print(sentence_corpus)
print("stories: "+str(len(story_corpus)))
#print("story climaxes: "+str(len(story_corpus)))
print("paragraphs: "+str(len(paragraph_corpus)))
print("sentences: "+str(len(sentence_corpus)))
#print(story_lump[0])
print()

print("mythos references")
for mythos_ref in mythos_refs.keys():
    print(mythos_ref)
    print(mythos_refs[mythos_ref])
print()


vectorizer = CountVectorizer(stop_words="english")
sentence_tf = vectorizer.fit_transform(sentence_corpus)
vocab = vectorizer.get_feature_names()
vocabn = numpy.array(vocab)
#print(vocab)
#print(vectorizer.get_feature_names())
#print(sentence_vec.toarray())

all_tf = vectorizer.transform(story_lump)
setup_tf = vectorizer.transform(setup_lump)
climax_tf = vectorizer.transform(climax_lump)
paragraph_tf = vectorizer.transform(paragraph_corpus)
story_tf = vectorizer.transform(story_corpus)
setup_story_tf = vectorizer.transform(setup_corpus)
climax_story_tf = vectorizer.transform(climax_corpus)

sentence_idf = TfidfTransformer()
sentence_idf.fit(sentence_tf)
paragraph_idf = TfidfTransformer()
paragraph_idf.fit(paragraph_tf)
story_idf = TfidfTransformer()
story_idf.fit(story_tf)
setup_story_idf = TfidfTransformer()
setup_story_idf.fit(setup_story_tf)
climax_story_idf = TfidfTransformer()
climax_story_idf.fit(climax_story_tf)

print("all_tfidf_sentence")
all_tfidf_sentence = sentence_idf.transform(all_tf)
all_tfidf_sentence_array = all_tfidf_sentence.toarray()
inds = all_tfidf_sentence_array.argsort()
#print(inds[0])
topWords = vocabn[inds]
#topRanks = all_tfidf_sentence_array[0][inds]
#print(topRanks[0][-200:])
print(topWords[0][-20:])

print("all_tfidf_story")
all_tfidf_story = story_idf.transform(all_tf)
all_tfidf_story_array = all_tfidf_story.toarray()
inds = all_tfidf_story_array.argsort()
#print(inds[0])
topWords = vocabn[inds]
#topRanks = all_tfidf_story_array[0][inds]
#print(topRanks[0][-200:])
print(topWords[0][-20:])

print("climax_tfidf_sentence")
climax_tfidf_sentence = sentence_idf.transform(climax_tf)
climax_tfidf_sentence_array = climax_tfidf_sentence.toarray()
inds = climax_tfidf_sentence_array.argsort()
#print(inds[0])
topWords = vocabn[inds]
print(topWords[0][-20:])

print()
print("story_tfidf_story")
story_tfidf_story = story_idf.transform(story_tf)
story_tfidf_story_array = story_tfidf_story.toarray()
inds = story_tfidf_story_array.argsort()
topWords1 = vocabn[inds]
setup_tfidf_story = story_idf.transform(setup_story_tf)
setup_tfidf_story_array = setup_tfidf_story.toarray()
inds = setup_tfidf_story_array.argsort()
topWords2 = vocabn[inds]
climax_tfidf_story = story_idf.transform(climax_story_tf)
climax_tfidf_story_array = climax_tfidf_story.toarray()
inds = climax_tfidf_story_array.argsort()
topWords3 = vocabn[inds]
for (title,words1,words2,words3) in zip(story_titles,topWords1,topWords2,topWords3):
    print(title)
    print(words1[-15:])
    print(words2[-15:])
    print(words3[-15:])
    print()
