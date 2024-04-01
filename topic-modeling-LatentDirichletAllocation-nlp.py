import pandas as pd
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from wordcloud import WordCloud
from pprint import pprint
import pyLDAvis
pyLDAvis.enable_notebook()  # don't skip this
import matplotlib.pyplot as plt

font1 = r'./font/NotoSansTC-Regular.otf'

df = pd.read_csv('data.csv', encoding='utf-8-sig')

# spilt content by comma
df['content'] = df['content'].apply(lambda x: x.split(','))
content2dList = df['content'].tolist()

dictionary = corpora.Dictionary(content2dList)
texts = content2dList
corpus = [dictionary.doc2bow(text) for text in content2dList]

# Building LDA Model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                        id2word=dictionary,
                                        num_topics=4, # Identifies the 4 topic trends for transportation
                                        random_state=100,
                                        update_every=1,
                                        chunksize=100,
                                        passes=20,
                                        alpha='auto',
                                        per_word_topics=True)
doc_lda = lda_model[corpus]

pprint(lda_model.show_topics(formatted=False))

# Compute Perplexity
print('The Perplexity Score is : ', lda_model.log_perplexity(corpus))  # measures of how good the model is. The lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=content2dList, dictionary=corpora.Dictionary(content2dList), coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('The Coherence Score is : ', coherence_lda)

vis = pyLDAvis.gensim_models.prepare(lda_model, 
                                    corpus, 
                                    dictionary, 
                                    mds="mmds",
                                    R=15) # the number of word a topic should contain.
pyLDAvis.save_html(vis, 'lda_visualization.html')

cloud = WordCloud(background_color='white',
                prefer_horizontal=1,
                height=330,
                max_words=200,
                colormap='flag',
                collocations=True,
                font_path=font1)

topics = lda_model.show_topics(formatted=False)

fig, axes = plt.subplots(2,2 , figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    plt.imshow(cloud.fit_words(dict(lda_model.show_topic(i, 200))))
    plt.gca().set_title('Topic' + str(i), fontdict=dict(size=12))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.suptitle("The Top 4 Research Topic Trend", 
            y=1.05,
            fontsize=18,
            fontweight='bold'
            )
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()
