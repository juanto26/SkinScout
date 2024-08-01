'''
==========================================================================================================================================

Final Project | Skin-Scout

Batch        : FTDS-BSD-006
Group        : 3

Team members : 
- Achmad Abdillah Ghifari : Data Analyst
- Celine Clarissa         : Data Scientist
- Evan Juanto             : Data Engineer

HuggingFace      : https://huggingface.co/spaces/celineclarissa/Skin-Scout

Original Dataset : https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data

Team GitHub      : https://github.com/juanto26/p2-final-project-skinscout


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
import streamlit as st
import EDA
import Home
import Review_Classification
import Recommender_System

# Creating the navigation sidebar
navigation = st.sidebar.selectbox('Menu:', ('Home', 'EDA', 'Review Classification', 'Recommender System'))

# Running py files
if navigation == 'Home':
    Home.run()
elif navigation == 'EDA':
    submenu = st.sidebar.selectbox('Submenu:', 
                                  ['WordCloud', 'Recommendation by Year', 
                                   'Customer Characteristics', 'Price of Products', 
                                   'Distribution of Loves Count'])
    EDA.run(submenu)
elif navigation == 'Recommender System':
    Recommender_System.run()
else:
    Review_Classification.run()

# Creating the markdown in the sidebar
st.sidebar.markdown('# About Skin-Scout')
st.sidebar.write("Skin-Scout is an application that uses advanced Natural Language Processing (NLP) technology that is able to analyze your skincare product review to find out the your likelihood to recommend a certain product. Other than that, Skin-Scout can also give you skincare product recommendations based on the product you insert to the system.")

st.sidebar.markdown('# Dataset')
st.sidebar.write('The original dataset can be accessed [here](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews?select=reviews_0-250.csv).')

st.sidebar.markdown('# GitHub')
st.sidebar.write('The repository of our project can be accessed [here](https://github.com/juanto26/p2-final-project-skinscout).')

st.sidebar.markdown('# Main Contributors')
st.sidebar.write("Achmad Abdillah Ghifari - Data Analyst")
st.sidebar.write("Celine Clarissa - Data Scientist")
st.sidebar.write("Evan Juanto - Data Engineer")