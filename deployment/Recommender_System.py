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

# Import

## Import Libraries
import pandas as pd
import numpy as np

## Import for Feature Engineering and System Building
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

## Import for Deployment
import streamlit as st

# Load Data
df = pd.read_csv('finalproject_clean.csv')

# Feature Engineering

## Clean values in columns 'ingredients' and 'highlights' by making them in one list
df['ingredients'] = df['ingredients'].apply(lambda x: ' '.join(eval(x)) if isinstance(x, str) else '')
df['highlights'] = df['highlights'].apply(lambda x: ' '.join(eval(x)) if isinstance(x, str) else '')

## Join values in columns 'ingredients' and 'highlights' to column 'combined_text'
df['combined_text'] = df['highlights'] + ' ' + df['ingredients']

## Define vectorizer
tfidf = TfidfVectorizer(stop_words='english')

## Fit and transform text with vectorizer
tfidf_matrix = tfidf.fit_transform(df['combined_text'])

## Take sample of matrix
rng = np.random.default_rng(seed=42)
tfidf_matrix = np.random.rand(10000,10000).astype(np.float64)

# Create Recommendation System

## Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

## Define function to generate product recommendation
def get_recommendations_by_name(product_name, cosine_sim=cosine_sim, df=df, num_recommendations=5):
    '''
    This function is used to get product recommendation based on the product name that the user inserts.
    '''
        
    ### Find product index based on product name
    index = df[df['product_name'] == product_name].index[0]
    
    ### Calculate cosine similarity score between all products and inserted product
    sim_scores = list(enumerate(cosine_sim[index]))
    
    ### Sort products based on cosine similarity (except for the inserted product itself)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:]
    
    ### Get recommended product index
    item_indices = [i[0] for i in sim_scores]
    
    ### Create dataframe for recommended products
    recommendations = df.iloc[item_indices]
    
    ### Remove inserted product from recommendations
    recommendations = recommendations[recommendations['product_name'] != product_name]
    
    ### Drop duplicates based on column 'product_name'
    unique_recommendations = recommendations.drop_duplicates(subset='product_name')
    
    ### Return recommendations
    return unique_recommendations.head(num_recommendations)

# Define function for when file is running
def run():
    
    ## Show title
    st.title('Product Recommender')

    ## Create line
    st.markdown('---')

    ## Make form
    with st.form("Final_Project_Form"):

        ### Create short description
        st.write("Insert a product name here, and we will provide similar products! With Skin-Scout's Recommender System, we will give 5 product recommendations based on the product you insert here.")

        ### Define feature
        product_name = st.text_input(label='## What is the name of the product?', value='Lip Sleeping Mask Intense Hydration with Vitamin C', help='Enter the name of the product here.')

        ### Create submit button
        submitted = st.form_submit_button("Submit")

    ## Create condition
    if submitted:
        try:
            ### Get recommendations for inserted product name
            recommendations = get_recommendations_by_name(product_name)
    
            ### Clean recommendations
            recommendations = recommendations[['product_name','brand_name']].reset_index()
            recommendations = recommendations.rename(columns={'product_name': 'Product Name', 'brand_name': 'Brand Name'})
            recommendations.drop(columns='index',inplace=True)
    
            ### Print recommendations
            container = st.container(border=True)
            container.markdown('## Result')
            container.write(f"##### You like {product_name}. Therefore, we recommend you to try:")
            container.write(recommendations)
        except:
            ### Print recommendations
            container = st.container(border=True)
            container.markdown('## Result')
            container.write(f"##### Product is not available.")

        ### Show spinner after submitting
        st.spinner(text='Please wait for the result.')

# Execute file
if __name__ == '__main__':
    run()