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
model with an F1-Score of 80%. This is done by using model such as SVC and cosine similarity in order to create the model. By creating
this model, our objective is to make the process of finding the perfect skincare product more time-efficient and less frustrating.

==========================================================================================================================================
'''

# Import libraries
import datetime as dt
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import pandas as pd
import psycopg2 as db
from elasticsearch import Elasticsearch

# Function to query data from PostgreSQL
def queryPostgresql():
    # Connecting to PostgreSQL database
    conn_string="dbname='airflow' host='postgres' user='airflow' password='airflow'"
    conn=db.connect(conn_string)
    # Executing the query and saving the result to a CSV file
    df=pd.read_sql("select * from table_final",conn)
    df.to_csv('/opt/airflow/dags/finalproject_raw.csv',index=False)
    print("-------Data Saved------")

# Function to clean the data
def clean_data():
    # Reading the raw data from CSV
    df = pd.read_csv('/opt/airflow/dags/finalproject_raw.csv')
    # Dropping unnecessary columns
    df = df.drop(['primary_category', 'author_id', 'product_id',
                  'child_max_price', 'child_min_price', 'review_title'], axis=1)
    df = df.drop(['rating_y', 'product_name_y', 'brand_name_y', 'price_usd_y'], axis=1)
    # Renaming columns for consistency
    df = df.rename(columns={
        'product_name_x': 'product_name',
        'brand_name_x': 'brand_name',
        'rating_x': 'rating',
        'price_usd_x': 'price_usd'
    })
    # Filling missing values with median, mode, or default values
    df['is_recommended'] = df['is_recommended'].fillna(df['is_recommended'].median())
    df['helpfulness'] = df['helpfulness'].fillna(df['helpfulness'].median())
    df['skin_tone'] = df['skin_tone'].fillna(df['skin_tone'].mode()[0])
    df['eye_color'] = df['eye_color'].fillna(df['eye_color'].mode()[0])
    df['skin_type'] = df['skin_type'].fillna(df['skin_type'].mode()[0])
    df['hair_color'] = df['hair_color'].fillna(df['hair_color'].mode()[0])
    df['size'] = df['size'].fillna(df['size'].mode()[0])
    df['variation_type'] = df['variation_type'].fillna(df['variation_type'].mode()[0])
    df['variation_value'] = df['variation_value'].fillna(df['variation_value'].mode()[0])
    df['ingredients'] = df['ingredients'].fillna(df['ingredients'].mode()[0])
    df['highlights'] = df['highlights'].fillna(df['highlights'].mode()[0])
    df['tertiary_category'] = df['tertiary_category'].fillna(df['tertiary_category'].mode()[0])
    # Dropping duplicate rows
    df = df.drop_duplicates()
    # Dropping rows with any remaining missing values
    df = df.dropna()
    # Saving the cleaned data to a new CSV file
    df.to_csv('/opt/airflow/dags/finalproject_clean.csv',index=False)

# Function to insert data into Elasticsearch
def insertElasticsearch():
    # Creating an Elasticsearch client
    es = Elasticsearch() 
    # Reading the cleaned data from CSV
    df=pd.read_csv('/opt/airflow/dags/final_project_clean.csv')
    # Indexing data into Elasticsearch
    for i,r in df.iterrows():
        doc=r.to_json()
        res=es.index(index="final_project",doc_type="doc",body=doc)
        print(res)	

# Default arguments for the DAG
default_args = {
    'owner': 'Final_Project',
    'start_date': dt.datetime(2024, 7, 29),
    'retries': 10,
    'retry_delay': dt.timedelta(minutes=1),
}

# Defining the DAG
with DAG('MyDBdag',
         default_args=default_args,
         schedule_interval='30 6 * * *',      # '0 * * * *',
         ) as dag:
    # Task to query data from PostgreSQL
    getData = PythonOperator(task_id='QueryPostgreSQL',
                                 python_callable=queryPostgresql)
    # Task to clean the data
    cleandata = PythonOperator(task_id = 'DataCleaning',
                                 python_callable=clean_data)
    # Task to insert data into Elasticsearch
    insertData = PythonOperator(task_id='InsertDataElasticsearch',
                                 python_callable=insertElasticsearch)


# Setting task dependencies
getData >> cleandata >> insertData