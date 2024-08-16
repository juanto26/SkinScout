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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from io import BytesIO

# Setting up the page config
st.set_page_config(
    page_title = 'Skin-Scout | EDA',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Function to help generate the wordcloud
def generate_word_cloud(dataframe, title):
    text = ' '.join(dataframe['review_text'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS).generate(text)
    bytes = BytesIO()
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=14, pad=20)
    plt.axis('off')
    plt.savefig(bytes, format='png')
    bytes.seek(0)

    return bytes

# Function to create the customer bar chart
def customer_bar_chart(column_name, data):
    count_data = data[column_name].value_counts()

    plt.figure(figsize=(8, 6))
    plt.bar(count_data.index, count_data)
    plt.xlabel(column_name.replace('_', ' ').title())
    plt.ylabel(f'Number of Customers with {column_name.replace("_", " ").title()}')
    plt.title(f'barplot of {column_name.replace("_", " ").title()}')
    plt.xticks(rotation=45)
    st.pyplot(plt)

# Function to remove outliers
def remove_outliers(data, lower_quantile=0.25, upper_quantile=0.75):
    lower_bound = np.quantile(data, lower_quantile)
    upper_bound = np.quantile(data, upper_quantile)
    cleaned_data = data[(data >= lower_bound) & (data <= upper_bound)]
    return cleaned_data

# Function to run the streamlit
def run(submenu):
    # Making the title
    st.title('Exploratory Data Analysis of Skin- Scout')

    # Creating the path to the dataset
    data = pd.read_csv('finalproject_clean.csv')


    # Defining the submenu for EDA for the wordcloud
    if submenu == "WordCloud":
        st.write('### Wordcloud of Recommendation')
        wordcloud = st.selectbox('Categories', ['All Recommendations', 'Recommended by Users', 'Not Recommended by Users'])
        if wordcloud == 'All Recommendations':
            all_reviews_buf = generate_word_cloud(data, 'All Recommendations')
            st.image(all_reviews_buf, use_column_width=True)
        elif wordcloud == 'Recommended by Users':
            recommended = data[data['is_recommended'] == 1]
            recommended_buf = generate_word_cloud(recommended, 'Recommended by Users')
            st.image(recommended_buf, use_column_width=True)
        elif wordcloud == 'Not Recommended by Users':
            not_recommended = data[data['is_recommended'] == 0]
            not_recommended_buf = generate_word_cloud(not_recommended, 'Not Recommended by Users')
            st.image(not_recommended_buf, use_column_width=True)                   
        with st.expander("Insight"):
            st.write("1. For the wordcloud for all recommendation the top 5 words are (product, love, skin, use, and face). These words mean that Sephora's customers mostly talk about how they love the product and how the product is used for their skin and face.")
            st.write("2. For the wordcloud for customer recommending the product, the top 5 words are (product, love, skin, use, and using). We can see that customers that recommend skincare products sold at Sephora use positive words like love which emphasize how the customer like the product and also words like use and using to emphasize how they use the product because they like the product.")
            st.write("3. For the wordcloud for customer not recommending the product, the top 5 words are (product, skin, didnt, face, and dont). We can see that customers that does not recommend the skincare products sold at Sephora usually use negative words like didn't and don't which we could infer to them not using the product. Other than that, customer also use past tense words such as used which could infer that they no longer use the product due to them not recommending the product.")
            st.write("With this insight we can see that there are differences between words used in positive and negative recommendations. Positive recommendation tend to use positive words such as love and present tense like use while negative recommendation tend to use negative word such as dont and didnt. While also using past tense like used.")

    # Defining the submenu for EDA for the Linechart
    elif submenu == "Recommendation by Year":
        st.write('### Visualizing Recommendation by Year')
        data['submission_time'] = pd.to_datetime(data['submission_time'])
        data['submission_year']= data['submission_time'].dt.year
        Count = data.groupby(['submission_year', 'is_recommended']).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 6))
        Count.plot(kind='line', ax=ax)
        ax.legend(['Not Recommended', 'Recommended'], title='Recommendation')
        st.pyplot(fig)
        with st.expander("Insight"):
            st.write('1. The number of recommendation from customers is significantly higher compared to non-recommendation throught the year 2017 to 2023.')
            st.write('2. Both recommendation and non-recommendation have the same trend peaking at 2020 and declining afterward. The main difference is the increase and decrease from recommendation is signficantly steeper than non-recommendation.')
            st.write('With this insight we can see that both recommendation and non-recommendation peaked at the year 2020 before declining. This might happen due to the pandemic leading people to buy less Sephora products leading to the decline of recommendation and non-recommendation number.')

    # Defining the submenu for EDA for the customer characteristics
    elif submenu == "Customer Characteristics":
        st.write('### Characteristics of Customers')
        characteristics = st.selectbox('Categories', ['skin_tone', 'eye_color', 'skin_type', 'hair_color'])
        if characteristics == 'skin_tone':
            customer_bar_chart('skin_tone', data)
            with st.expander("Skin Tone Insight"):
                st.write('From the skin tone we can see that the skin tone light is the most common while dark is the least common apart from not sure. From the bar chart we can see that there are a diverse range of 12 skin tone in the dataset although a major amount of data is centered around light, fair, and light-medium.')
        elif characteristics == 'eye_color':
            customer_bar_chart('eye_color', data)
            with st.expander("Eye Color Insight"):
                st.write('From the eye color we can see that the most common eye color is brown while the least common eye color is gray or grey. We can see that the data is diverse with 5 eye color with a majority of the data centered around brown with more than 25.000 data.')
        elif characteristics == 'skin_type':
            customer_bar_chart('skin_type', data)
            with st.expander("Skin Type Insight"):
                st.write('From the skin type we can see that the skin type of combination while the skin type oily is the least common although barely. We can see the data have 4 skin type with a majority of the data centered around combination skin type.')
        elif characteristics == 'hair_color':
            customer_bar_chart('hair_color', data)
            with st.expander("Hair Color Insight"):
                st.write('From the hair color we can see that the most common hair color is brown with gray being the least common. We can see that the data is varied with 7 hair color with a majority of the data having brown, blonde, and black hair color.')        
        with st.expander("General Insight"):
            st.write('With this insight we can infer that the most common characteristics from customers is a customer with light skin having a brown hair and eyes and having a combination of dry and oily skin. This could happen due to perhaps our marketing targeted these characteristics more compared to other type of characteristics.')

    # Defining the submenu for EDA for the top 10 cheapest and most expensive product
    elif submenu == "Price of Products":
        st.write('### Top 10 Products from Prices')
        cheap = data.groupby(['product_name', 'price_usd']).sum(numeric_only=True).reset_index().sort_values('price_usd', ascending=True).head(10).sort_values('price_usd', ascending=False)
        expensive = data.groupby(['product_name', 'price_usd']).sum(numeric_only=True).reset_index().sort_values('price_usd', ascending=False).head(10)
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
        ax1.barh(cheap['product_name'], cheap['price_usd'])
        ax1.set_title("TOP 10 Cheapest Products")
        ax1.set_xlabel("Price in Dollars")
        ax1.set_ylabel("Product Name")
        ax2.barh(expensive['product_name'], expensive['price_usd'])
        ax2.set_title("TOP 10 Most Expensive Products")
        ax2.set_xlabel("Price in Dollars")
        ax2.set_ylabel("Product Name")
        plt.suptitle("Top 10 Products Based on Price", fontsize=30)
        st.pyplot(fig)
        with st.expander("Insight"):
            st.write('1. From the top 10 cheapest price, we can see that the price in dollar ranges from 3 to 5 dollars. These products mostly consist of common skincare items like masks, tea, wipes, and papers. The cheapest product according to the product price consists of cleansing & exfoliating wipes and clean charcoal nose strip.')
            st.write('2. From the top 10 most expensive price, we can see that the price in dollar range from around 360 to 460 dollars. These product mostly consists of fancier skincare product like moisturizers, wands, serums, and even electronic devices for skincare. The most expensive product according to the product price is the DRx SpectraLite BodyWare Pro.')
            st.write('With this insight we can see that the skincare products at Sephora has clearly been seperated into two categories, more affordable products like mask, tea, and wipes which is around 3 to 5 dollars and more expensive skincare options such as moisturizer and electronic device for skincare ranging from 360 to 460 dollars. These two categories can be used in order to cater marketing campaign to certain customer demographic.')

    # Defining the submenu for EDA for the most favorited product
    elif submenu == "Distribution of Loves Count":
        st.write('### Number of Favorites that a Product Has')
        cleaned_data = remove_outliers(data['loves_count'], lower_quantile=0.25, upper_quantile=0.75)
        min_value = cleaned_data.min()
        q1d = cleaned_data.quantile(0.25)
        q2d = cleaned_data.quantile(0.5)
        q3d = cleaned_data.quantile(0.75)
        max_value = cleaned_data.max()
        iqr = q3d - q1d
        plt.figure(figsize=(6, 4))
        plt.boxplot(cleaned_data, vert=False, patch_artist=True)
        plt.annotate(f'Q1 = {q1d}', xy=(q1d, 1.06), xytext=(q1d, 1.2), arrowprops=dict(facecolor='black', arrowstyle='->'))
        plt.annotate(f'Q2 = {q2d}', xy=(q2d, 1.06), xytext=(q2d, 1.3), arrowprops=dict(facecolor='black', arrowstyle='->'))
        plt.annotate(f'Q3 = {q3d}', xy=(q3d, 1.06), xytext=(q3d, 1.2), arrowprops=dict(facecolor='black', arrowstyle='->'))
        plt.annotate(f'IQR = {iqr}', xy=((q1d+q3d)/2, 0.85), ha='center')
        plt.axvline(q1d, linestyle='--', color='gray', label='Q1')
        plt.axvline(q3d, linestyle='--', color='gray', label='Q3')
        plt.axvspan(q1d, q3d, alpha=0.2, color='gray', label='IQR')
        plt.xlabel('Product Loves Count')
        plt.title('Box Plot of Product Loves Count')
        st.pyplot(plt)
        with st.expander("Insight"):
            st.write('1. From the iqr we can see there are a significant spread of product being favorited (love count) with a spread from Q1 and Q3 of 46,656 love count.')
            st.write('2. From the Q1 we can see that 25% of the favorited product have a love count of 30,183.')
            st.write('3. From the Q2 we can see that 50% of the favorited product have a love count of 49,032.')
            st.write('4. From the Q3 we can see that 75% of the favorited product have a love count of 76,840.')
            st.write('5. From the whiskers we can see that the love count is positively skewed meaning that most of the products have lower love count.')

# Code to run the program
if __name__ == '__main__':
    run()