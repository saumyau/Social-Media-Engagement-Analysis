from tweepy import OAuthHandler, StreamListener
import tweepy
import json
import pandas as pd
import copy
import nltk
import numpy as np
import re
from bokeh.io import show, output_file
from bokeh.plotting import figure
from math import pi
from operator import itemgetter
from collections import OrderedDict
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from nltk import pos_tag
from nltk import ngrams
from tkinter import *
from tkinter import messagebox  
from tkinter import filedialog
from ttkthemes import themed_tk as tk
import tkinter.font as font
import seaborn as sns
import pandas as pd
import random
import PIL
from PIL import Image
from PIL import ImageTk
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import sys
import csv
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

global listUsers_df
global screen_name
listUsers_df = []


class StdOutListener(StreamListener):

    def on_data(self, data):
        print(data)
        return True

    def on_error(self, status):
        print(status)




# function to authenticate the credentials to connect to twitter using Tweepy API
def authentication():
    auth = tweepy.OAuthHandler('XX','XX')
    auth.set_access_token('XX','XX')
    api=tweepy.API(auth,wait_on_rate_limit=True)
    return api


# In[ ]:


api = authentication()


# In[ ]:


# 1. Imports data from the 'Categories_List.csv' contains the categorical data into a dataframe
# 2. Imports data from the 'Users_Set_1.csv' contains the list of user names into a dataframe
def load_files():
    try:
        global category_df
        cat_df = pd.read_csv('Categories_List.csv',encoding='ISO-8859-1')
        #print(category_df)
        category_df = pd.DataFrame()
        for column in cat_df.columns:
            col_count = 0
            column_data = cat_df[column].str.lower()
            category_df.insert(loc=col_count,column=column,value=pd.Series(column_data))
            col_count+=1
        #initalise user data set frame
    except:
        print("Please load the categories and user data set...")
    


# In[ ]:


# function to write data into a CSV file
def writeToCsv(name,df):
    df.to_csv(''+name+'.csv',encoding='utf-8',index=False)


# In[ ]:


# function to remove non-ASCII characters from a text
def removeNonAscii(text):
    return "".join(char for char in text if ord(char) < 128)


# In[ ]:


# function to fetch tweet results using cursors
def limit_handled(cursor):
    while True:
        try:
            yield cursor.next()
        except tweepy.RateLimitError:
            print("Time Limit. Sleeping now....")
            time.sleep(15 * 60)
            print("Trying again")


# In[ ]:


# 1. function to extract tweets,retweets & favorites of a given user
# 2. Data cleansing like - removing unneccessary words from the tweets are performed
# 3. Results are stored in the dataframe.
# 4. A CSV file of the fotmat - 'ScreeName.csv' will be exported containing a list of tweets, retweets and favorites.
def extractTweets(screen_name):
    #create empty data frame
    dataFrame = pd.DataFrame()
    tweet_count=0
    retweet_count=0
    fav_count=0
    tweet_text=[] # list of tweets
    retweet_text=[] # list of retweets
    fav_text=[] # list of favorite tweets
    loc_tweet =[]
    for tweets in limit_handled(tweepy.Cursor(api.user_timeline,screen_name=screen_name,count=200,tweet_mode='extended').items()):
        #print("Created at:%s, tweet is : %s " %(tweets.created_at,tweets.text))
        if (tweets.lang == "en"):
            raw_text = tweets.full_text
            text = re.sub(r"http\S+","",raw_text)
            tweet_text.append(removeNonAscii(text))
            tweet_count+=1
        if hasattr(tweets,'retweeted_status'):
            if (tweets.retweeted_status.lang == "en"):
                raw_text = tweets.retweeted_status.full_text
                text = re.sub(r"http\S+","",raw_text)
                retweet_text.append(removeNonAscii(text))
                retweet_count+=1
        if hasattr(tweets,'country'):
            loc_tweet.append(tweets.country)
        if(tweet_count == 2980 or retweet_count == 2980):
            break
    for fav_tweets in limit_handled(tweepy.Cursor(api.favorites,id=screen_name,tweet_mode='extended').items()):
        if(fav_tweets.lang  == 'en'):
            raw_text = fav_tweets.full_text
            text = re.sub(r"http\S+","",raw_text)
            fav_text.append(removeNonAscii(text))
            fav_count+=1
        if (fav_count > 200):
            break
    #print(tweet_count)
    # insert tweets in the dataframe
    dataFrame.insert(loc=0,column='Tweets',value=pd.Series(tweet_text)) 
    # insert retweets in the dataframe
    dataFrame.insert(loc=1,column='Retweets',value=pd.Series(retweet_text))
    #insert favorite tweets in the dataframe
    dataFrame.insert(loc=2,column='Favorite Tweets',value= pd.Series(fav_text))
    #print(dataFrame)
    dataFrame.insert(loc=3,column='Location',value=pd.Series(loc_tweet))
    writeToCsv(screen_name,dataFrame)
    performSentimentAnalysis(screen_name,dataFrame)


# In[ ]:


# 1.function to perform sentioment analysis of all the tweets
def performSentimentAnalysis(screen_name,dataFrame):
    df = copy.deepcopy(dataFrame)
    senti_df = df['Tweets'].append([df['Retweets'],df['Favorite Tweets']]).reset_index(drop=True)
    senti_df = pd.DataFrame(senti_df,columns=['twitter_text']).dropna()
    senti_df['category'] = np.nan
    senti_df = senti_df.reset_index(drop=True) 
    analyseSentiment = SentimentIntensityAnalyzer()
    for itr in range(0,len(senti_df)):
        index = list(np.where(senti_df['twitter_text'] == senti_df.iloc[itr]['twitter_text'] )[0])
        senti_score = analyseSentiment.polarity_scores(senti_df.iloc[itr]['twitter_text'])
        #classify pos or neg category based on the compound score
        if (senti_score['compound'] > 0.0):
            senti_df.loc[index,'category'] = 'positive'
        else:
            senti_df.loc[index,'category'] = 'negative'        
    writeToCsv(screen_name + '_Sentiment',senti_df)
    #print(senti_df)
    analyseText(senti_df,screen_name)   


# In[ ]:


# 1.function that takes the positive tweets, performs data cleansing and noun related words are extracted.
def analyseText(senti_df,screen_name):
    dataFrame = copy.deepcopy(senti_df)
    index = list(np.where(dataFrame['category'] == 'positive')[0])
    positive_text=[]
    all_text=[]
    all_nouns_text=[]
    for index_count in range(0,len(index)):
        positive_text.append(dataFrame.iloc[index_count]['twitter_text'])
    # set the stop words
    stop_words = set(stopwords.words('English'))
    #set the stemmer
    #stemmer = PorterStemmer()
    for itr in range(0,len(positive_text)):
        #tokenize
        tokens = word_tokenize(positive_text[itr])
        #remove stop words & punctuation(other characters)
        filtered_text = [ word.lower() for word in tokens if (word not in stop_words) and (word.isalpha()) and (len(word) > 1)]
        #print(filtered_text)
        # get list of all tokens
        all_text+=filtered_text
    #print(all_text)
    # resort to all stemmed words
    #all_stemmed_text = [stemmer.stem(word) for word in all_text]
    #print(all_stemmed_text)
    # take only the nouns
    tags = pos_tag(all_text)
    for word,pos in tags:
            if pos in ['NN','NNS','NNP','NNPS']:
                all_nouns_text.append(word)
    #print(all_nouns_text)
    getTopWords(all_nouns_text,screen_name)


# In[ ]:


# 1. function that gets the highest frequency words from the unigram, bigrams and trigrams list.
def getTopWords(all_nouns_text,screen_name):
    nouns_text = copy.deepcopy(all_nouns_text)
    topCount_df=pd.DataFrame()
    all_single=[]
    all_bigrams=[]
    all_trigrams=[]
    # for single words
    freq_single = Counter(nouns_text)
    for token,count in freq_single.most_common(10):
        all_single.append(token)
    # for bigrams
    bigrams = list(ngrams(nouns_text,2))
    freq_bi = Counter(bigrams)
    for token,count in freq_bi.most_common(10):
        all_bigrams.append(list(token))
    #for trigrams
    trigrams = list(ngrams(nouns_text,3))
    freq_tri = Counter(trigrams)
    for token,count in freq_tri.most_common(10):
        all_trigrams.append(list(token))
    # add single word top counts to the dataframe
    topCount_df.insert(loc=0,column='Single',value=pd.Series(all_single))
    # add bigrams word top count to the dataframe
    topCount_df.insert(loc=1,column='Bigrams',value=pd.Series(all_bigrams))
    # add trigrams word top count to the dataframe
    topCount_df.insert(loc=2,column='Trigrams',value=pd.Series(all_trigrams))
    #print(topCount_df)
    writeToCsv('top_count',topCount_df)
    findCategories(topCount_df,screen_name)   


# In[ ]:


# 1. function that keeps track of the number of words belongs to what categories.
def findCategories(topCount_df,screen_name):
    count_df = copy.deepcopy(topCount_df)
    # create dictionaries for tracking word cout & categories
    word_count_dict = {}
    word_category_dict = {}
    for row_itr in range(0,len(count_df)):
        for column_itr in range(0,len(count_df.columns)):
            key = count_df.iloc[row_itr][column_itr]
            if isinstance(key,str):
            #check if this key is already present in the dictionary
                if key not in word_count_dict.keys():
                    word_count_dict[key] = 0
                    word_category_dict[key] = []
            elif isinstance(key,list):
                for key_list in key:
                    if key_list not in word_count_dict.keys():
                        word_count_dict[key_list] = 0  
                        word_category_dict[key_list] = [] 
    #print(word_count_dict)
    # update count values in word_count_dict & categories in word_category_dict
    column_labels = list(category_df.columns)
    for key in word_count_dict.keys():
        for column in column_labels: 
            if any(category_df[column] == key):
                word_count_dict[key] +=1 
                word_category_dict[key].append(column)
    #print(word_count_dict)
    #print(word_category_dict)
    #sorted_word_count_dict = OrderedDict(sorted(word_count_dict.items(),key = itemgetter(1),reverse = True)[:5])
    #print(sorted_word_count_dict)
    #categoryWisePercent(sorted_word_count_dict,word_category_dict)
    categoryWisePercent(word_count_dict,word_category_dict,screen_name)
    


# In[ ]:


# 1. function that classifies words belonging to each categories and caluclates the categorywise percentage
def categoryWisePercent(word_count_dict,word_category_dict,screen_name):
    sorted_word_dict = copy.deepcopy(word_count_dict)
    category_dict = copy.deepcopy(word_category_dict)
    # create empty dataframe
    percent_df = pd.DataFrame(columns = ['Category','Words','Percentage'])
    # categorize it
    category_list= []
    for key in sorted_word_dict:
        cat_list = list (category_dict[key])
        for itr in range(0,len(cat_list)):
            if cat_list[itr] not in category_list:
                category_list.append(cat_list[itr])
    percent_df['Category'] = category_list
    percent_df['Words'] = 0
    percent_df['Percentage'] = 0 
    #fill words column
    for key in sorted_word_dict:
        cat_list = list(category_dict[key])
        for itr in range(0,len(cat_list)):
            index = list(np.where(percent_df['Category'] == cat_list[itr])[0])
            percent_df.at[index,'Words'] = percent_df['Words'][index] +1
    # fill percentage column
    total_word_count=0
    for row_count in range(0,len(percent_df)):
        total_word_count += percent_df['Words'][row_count]
    for row_count in range(0,len(percent_df)):
        percent_df.at[row_count,'Percentage'] = ( percent_df['Words'][row_count] / total_word_count) * 100
    percent_df.sort_values(['Percentage'],ascending = False, inplace = True)
    percent_df = percent_df.reset_index(drop = True)
    print(percent_df.to_string())
    listUsers_df.append(percent_df)
    plot_piechart(percent_df,screen_name)
    get_hashtags(screen_name)


def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d} g)".format(pct, absolute)


def on_key_event(event):
    print('you pressed %s' % event.key)
    key_press_handler(event, canvas, toolbar)


def get_hashtags(screen_name):
    hashtags_dict = {}
    l = StdOutListener()
    tweets = api.user_timeline(screen_name=screen_name, count=200)

    for tweet in tweets:
        hashtags = tweet.entities.get('hashtags')
        for hashtag in hashtags:
            if hashtag['text'] in hashtags_dict.keys():
                hashtags_dict[hashtag['text']] += 1
            else:
                hashtags_dict[hashtag['text']] = 1

    z = []
    for val in hashtags_dict.values():
        z.append(val)
    print(z)
    u = []
    for tag in hashtags_dict.keys():
        u.append(tag)
    print(hashtags_dict)
    print(sorted(hashtags_dict, key=hashtags_dict.get, reverse=True)[:10])
    x = range(len(hashtags_dict))
    y = []
    for i in range(len(hashtags_dict)):
        y.append(random.randint(0,len(hashtags_dict)))

    df = pd.DataFrame(dict(x=x, y=y, z=z, users=u))
    fig, ax = plt.subplots(figsize=(10, 15),facecolor='w')

    for key, row in df.iterrows():
        ax.scatter(row['x'], row['y'], row['z']*1000, alpha=.5)
        ax.annotate(row['users'], xy=(row['x'], row['y']))

    plt.xlabel("the X axis")
    plt.ylabel("the Y axis")
    plt.title("Hashtag Analytics")
    plt.savefig('bubble.png',bbox_inches='tight')
    #plt.show()
    #canvas = FigureCanvasTkAgg(fig, master=root)
    #canvas.show()
    #canvas.get_tk_widget().pack(side=Tk.RIGHT, fill=Tk.NONE, expand=1)
    #toolbar = NavigationToolbar2TkAgg(canvas, root)
    #toolbar.update()
    #canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.NONE, expand=1)
    #canvas.mpl_connect('key_press_event', on_key_event)




def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate
    


# In[ ]:
def plot_piechart(percent_df,screen_name):
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(aspect="equal"))
    per_df = copy.deepcopy(percent_df)
    max_value= max(list(per_df['Words']))
    index = list(np.where(per_df['Words'] == max_value)[0])
    explode_list = []
    for itr in range(0,len(per_df)):
        if itr in index:
            explode_list.append(0.1)
        else:
            explode_list.append(0.0)        
    #print(explode_list)
    explode_tuple = tuple(explode_list) 
    labels = list(per_df['Category'])
    values = list(per_df['Words'])
    wedges, texts, autotexts = ax.pie(values, autopct=lambda pct: func(pct, values),textprops=dict(color="w"))
    ax.legend(wedges, labels,title="Interests",loc="center left",bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(autotexts, size=8, weight="bold")
    ax.set_title("User Interest Piechart")
    plt.savefig('pie.png',bbox_inches='tight')
    #plt.show()
    #canvas = FigureCanvasTkAgg(fig, master=root)
    #canvas.show()
    #canvas.get_tk_widget().pack(side=Tk.RIGHT, fill=Tk.NONE, expand=1)
    #toolbar = NavigationToolbar2TkAgg(canvas, root)
    #toolbar.update()
    #canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.NONE, expand=1)
    #canvas.mpl_connect('key_press_event', on_key_event)

# 1. function that classifies users belonging to each category.
# 2. 'User_Categorized_List.csv' contains the list of users against different categories.


# In[ ]:


# 1 . function to browse for the user dataset file.
def browseFile():
    global filename
    if textbox.compare("end-1c","!=","1.0"):
        textbox.delete(1.0,END)
    if lb.index('end') != 0:
        lb.delete(0,END)        
    filename = filedialog.askopenfilename(initialdir="/",title="Select File",filetypes = 
                                         (("CSV Files","*.csv"),("All Files","*.*"))
                                         )
    textbox.insert(END,filename)
    col_name =['users']
    data = pd.read_csv(filename, names = col_name)
    item_list=data.users.tolist()
    print(item_list)
    #lbl = Listbox()
    for i in item_list:
        lb.insert(END, i)
    lb.place(x=150,y=200,width=300)
    lb.bind("<<ListboxSelect>>", onSelect)    

def onSelect(val):
    sender = val.widget
    idx = sender.curselection()
    value = sender.get(idx)
    var = StringVar()
    var.set(value)
    print(type(value))
    load_files(filename)
    #main()
    extractTweets(value)
         

def retrieve_input():
    inputValue = textBox.get("1.0","end-1c")
    if lb1.index('end') != 0:
        lb1.delete(0,END)
    if inputValue != "":
        lb1.insert(1,"Execution Started")
        load_files()
        lb1.insert(2,"Bubble chart and pie chart generation")
        #main()
        extractTweets(inputValue)
        lb1.insert(3,"Execution Completed")
        msg = messagebox.showinfo('Success','Results Generated Successfully. ')  
    else:
        msg = messagebox.showinfo('Error','Please browse and select the user dataset file')


    
# In[ ]:
def show_pie():
    im = PIL.Image.open("pie.png")
    photo = PIL.ImageTk.PhotoImage(im)

    label = Tk.Label(root, image=photo)
    label.image = photo  # keep a reference!
    label.place(x=500,y=0)

def show_bubble():
    novi = Toplevel()
    canvas = Canvas(novi, width = 1000, height = 1000)
    canvas.pack(fill = BOTH,expand=1)
    gif1 = PhotoImage(file = 'bubble.png')
                                #image not visual
    canvas.create_image(0, 0, image = gif1, anchor = NW)
    #assigned the gif1 to the canvas object
    canvas.gif1 = gif1
    #canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.NONE, expand=1)
    #canvas = FigureCanvasTkAgg(fig, master=root)
    #canvas.show()
    #canvas.get_tk_widget().pack(side=Tk.RIGHT, fill=Tk.NONE, expand=1)
    #toolbar = NavigationToolbar2TkAgg(canvas, root)
    #toolbar.update()
    #canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.NONE, expand=1)
    #canvas.mpl_connect('key_press_event', on_key_event)

def show_data():
    with open("Categories_List.csv", newline = "") as file:
        reader = csv.reader(file)
        r = 0
        for col in reader:
            c = 0
            for row in col:
             # i've added some styling
                label = Tk.Label(root, width = 10, height = 2, text = row, relief = Tk.RIDGE)
                label.grid(row = r, column = c)
                c += 1
            r += 1

# UI elements
#root= tkinter.Tk()
root = Tk.Tk()
#root = ttk.Tkinter()
#root= tk.ThemedTk()
#root.get_themes()
root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
root.focus_set() 
root.bind("<Escape>", lambda e: e.widget.quit())
root.configure(background='#a52a2a')
#root.style = ttk.Style()
text_label = Label(root,text='Enter User Id', height=2, width=12)
#text_label.config(font=("Helvetica", 15))
text_label.place(x=150,y=100)
textBox=Text(root, height=2, width=15)
textBox.place(x=300,y=100)
buttonCommit=Button(root, height=1, width=10, text="Execute",command=lambda: retrieve_input())
buttonCommit.place(x=150,y=160)
root.title('Interest Analysis Tool')
title_label = Label(root,text='Interest Analysis Tool')
title_label.place(x=580,y=10)
title_label.config(font=("Cambria", 23))
filename=""
button = Button(root, height=1, width=15, text="Show Pie Chart",command=lambda: show_pie())
button.place(x=150,y=400)
button1 = Button(root, height=1, width=15, text="Show Bubble Chart",command=lambda: show_bubble())
button1.place(x=150,y=440)
#button2 = Button(root, height=1, width=15, text="View Data Set",command=lambda: show_data())
#button2.place(x=150,y=480)
button = Tk.Button(master=root, text='Quit',width=10, command=_quit)
button.place(x=150,y=580)
button.config(font=("Cambria", 12))
lb1= Listbox(root)
lb1.place(x=150,y=200,width=300)
#Tk.mainloop()
root.mainloop()

