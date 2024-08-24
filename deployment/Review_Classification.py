'''
==========================================================================================================================================

Final Project | Skin-Scout

Batch        : FTDS-BSD-006
Group        : 3

Team members : 
- Achmad Abdillah Ghifari : Data Analyst
- Celine Clarissa         : Data Scientist
- Evan Juanto             : Data Engineer

Hugging Face     : https://huggingface.co/spaces/celineclarissa/Skin-Scout

Original Dataset : https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data

Team GitHub      : https://github.com/FTDS-assignment-bay/p2-final-project-ftds-006-bsd-group-006


Background

The current skincare market is flooded with countless products each with unique ingredients and highlights. Consumers often struggle to
decide which product most consumer recommend due to the large amount of reviews for each different products, making reading to all the
review traditionally wasting too much time and effort. While other metrics such as star rating is present on most skincare website,
relying on only star rating to rate the quality of a product is unreliable as research has shown that star rating has many problem such as
negativity bias where one negative aspect could lead to users leading a low star despite excelling in other area and also sometime the
review and star a user give has discreptancy with some research finding only a moderate correlation between review and star rating. Hence,
consumers are left to go through multiple reviews in order to get an accurate insight regarding certain skincare product. Due to this
factor our teams goal is to create an application where we could make this process easier by finding out whether a certain user will
recommend or not recommend a product based on their review.


Problem Statement and Objective

We want to create an application that utilizes Natural Language Processing (NLP) and a recommender system in order to help predict
whether a customer will recommend a product or not and also to give recommendation of similar skincare product. Our goal is to create a
model with an F1-Score of 80%. This is done by using model such as SVC and cosine similarity in order to create the model. By creating
this model, our objective is to make the process of finding the perfect skincare product more time-efficient and less frustrating.

==========================================================================================================================================
'''

# Import libraries
import pandas as pd
import re
import pickle

# Import for Feature Engineering
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Import for Deployment
import streamlit as st

# Load model
with open('Final_Project_Model_NLP.pkl', 'rb') as file_1:
    Final_Project_Model_NLP = pickle.load(file_1)

# Text preprocessing

## Define stopwords
stopwords_eng = stopwords.words('english')

## Add words that frequently occur in all skincare reviews to stopwords
stopwords_eng.append('face')
stopwords_eng.append('product')
stopwords_eng.append('skin')
stopwords_eng.append('use')
stopwords_eng.append('using')
stopwords_eng.append('used')
stopwords_eng.append('really')

## Create text preprocessing function
def text_preprocessing(text):
    '''
    This function is created to do text preprocessing: change text to lowercase, Remove numbers and punctuation symbols, Remove stopwords,
    lemmatize text, and tokenize text. Text preprocessing can be done just by calling this function.
    '''
    ### Change text to lowercase
    text = text.lower()
    ### Remove numbers
    text = re.sub(r'\d+', '', text)
    ### Remove comma
    text = text.replace(',', '')
    ### Remove period symbol
    text = text.replace('.', '')
    ### Remove exclamation mark
    text = text.replace('!', '')
    ### Remove question mark
    text = text.replace('?', '')
    ### Remove quotation mark
    text = text.replace('"', '')
    text = text.replace("'", '')
    text = text.replace('’', '')
    ### Remove hyphen
    text = text.replace('-', ' ')
    text = text.replace('—', ' ')
    ### Remove ampersand
    text = text.replace('&', 'and')
    ### Remove whitespace
    text = text.strip()
    ### Tokenization
    tokens = word_tokenize(text)
    ### Remove stopwords
    tokens = [word for word in tokens if word not in stopwords_eng]
    ### Lemmatization: minimize words with same or similar meaning
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    ### Combine tokens
    text = ' '.join(tokens)
    ### Return processed text
    return text

# Create class dictionary
dict_class = {0: 'not recommended',
              1: 'recommended'}

# Define function for when file is running
def run():
    
    ## Show title
    st.title('Skincare Review')

    ## Create line
    st.markdown('---')

    ## make form
    with st.form("Final_Project_Form"):

        ### define each feature
        product_name = st.text_input(label='## What is the name of the product?', 
                                     help='Enter the name of the product here.', 
                                     value='Facial Treatment Essence (Pitera Essence)')
        brand_name = st.text_input(label='## What is the brand name of the product?', 
                                   help='Enter the brand of the product here.', 
                                   value='SK-II')
        review_text = st.text_area(label='## How was the product?', 
                                   help='Write your review here.', 
                                   value="Honestly, my skin feels more rejuvenated within my first week of using SK-II's Pitera Facial Treatment Essence than it has all year, which is a complete win in my book. The brand claims that their essence will help your skin look and feel even brighter within 28 days, so I'll be continuing my use of this product to see the results for myself. Although SK-II's essence is priced higher than others on the market, the quality of their ingredients (and over 40 years of expertise in their field) lets them speak for themselves. If you're looking for a step to add to your skincare routine that is sure to revive your skin, no matter what type, look no further than SK-II")

        ### Create submit button
        submitted = st.form_submit_button("Submit")

    ## Define inference data based on inputted data
    inf_data = {
    'Product Name': product_name,
    'Brand Name': brand_name,
    'Review Text': review_text
}

    ## Make dataframe for inference data
    inf_data = pd.DataFrame([inf_data])

    ## Create condition
    if submitted:

        ### Preprocess text using function
        inf_data['text_processed'] = inf_data['Review Text'].apply(lambda x: text_preprocessing(x))

        ### Define result using model
        y_pred_inf = Final_Project_Model_NLP.predict(inf_data.text_processed)

        ### Print result
        container = st.container(border=True)
        container.markdown('## Result')
        container.write(f'##### Based on your review, the product is {dict_class[int(y_pred_inf)]}.')

        ### Show spinner after submitting
        st.spinner(text='Please wait for the result.')

# Execute file
if __name__ == '__main__':
    run()