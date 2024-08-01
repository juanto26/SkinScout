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

Team GitHub      : https://github.com/juanto26/p2-final-project-skinscou


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
model with an accuracy of 80%. This is done by using model such as svc and cosine similarity in order to create the model. By creating
this model, our objective is to make the process of finding the perfect skincare product more time-efficient and less frustrating.

==========================================================================================================================================
'''

# Import libraries
import streamlit as st

# Creating the page config
st.set_page_config(
    page_title= 'Skin-Scout | Home',
    layout= 'wide',
    initial_sidebar_state='expanded'
)

# Creating a function to run the program
def run():
    
    # Loading image and markdown
    st.image('SkinScout_Logo.png', width=800)
    st.markdown('<p style="font-size: 50px; color: #000000; font-weight: bold; text-align: center;">Discover the Needs of Your Skin</p>', unsafe_allow_html=True)

    st.markdown('---')

    # Creating container for home-page description
    container = st.container(border=True)
    container.markdown('<h1 style="font-size: 30px;">Welcome to Skin-Scout</h1>', unsafe_allow_html=True)
    container.write("Skin-Scout is an application that helps in discovering customers' skincare preferences. By using advanced Natural Language Processing (NLP) technology, Skin-Scout helps elevate how you could find recommendations and goes beyond traditional review analysis by offering a deeper understanding regarding your skincare choice.")

    container1 = st.container(border=True)
    container1.markdown('<h1 style="font-size: 30px;">Why Choose Skin-Scout?</h1>', unsafe_allow_html=True)
    container1.write('Skin-Scout offers many innovative features to help you unlock the full potential of skincare products. The following are what you can get by using our model.')
    container1.markdown('<h1 style="font-size: 20px;">User Review Analysis</h1>', unsafe_allow_html=True)
    container1.write('Skin-Scout analyze user reviews in order to gauge the likelihood that the users will recommend the product with remarkable accuracy. Skin-Scout helps provide valuable insight on whether a customer will likely recommend the product based on their reviews')
    container1.markdown('<h1 style="font-size: 20px;">Personalized Suggestion</h1>', unsafe_allow_html=True)
    container1.write('Skin-Scout also have a feature to suggest similar product to the customer according to the products ingredients and highlights. Just enter your preferred product and let our model do the rest.')

    container2 = st.container(border=True)
    container2.markdown('<h1 style="font-size: 30px;">How to use Skin-Scout?</h1>', unsafe_allow_html=True)
    container2.markdown('<h1 style="font-size: 20px;">User Review Analysis</h1>', unsafe_allow_html=True)
    container2.write('Provide information regarding a customers review or try to make your own, then our model will calculate whether you will recommend the product or not based on your review')
    container2.markdown('<h1 style="font-size: 20px;">Personalized Suggestions</h1>', unsafe_allow_html=True)
    container2.write('Provide information regarding a product that the customer has recommended, then our model will recommend similar product based on the product ingredients and highlight.')

# Code to run the program
if __name__ == '__main__':
    run()