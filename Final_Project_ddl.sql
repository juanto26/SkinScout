/*
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
*/

-- Create new table
CREATE TABLE table_final (
    "author_id" VARCHAR(10000),
    "rating_x" INT,
    "is_recommended" FLOAT,
    "helpfulness" FLOAT,
    "total_feedback_count" INT,
    "total_neg_feedback_count" INT,
    "total_pos_feedback_count" INT,
    "submission_time" VARCHAR(10000),
    "review_text" VARCHAR(10000),
    "review_title" VARCHAR(10000),
    "skin_tone" VARCHAR(10000),
    "eye_color" VARCHAR(10000),
    "skin_type" VARCHAR(10000),
	"hair_color" VARCHAR(10000),
	"product_id" VARCHAR(10000),
	"product_name_x" VARCHAR(10000),
	"brand_name_x" VARCHAR(10000),
	"price_usd_x" FLOAT,
	"product_name_y" VARCHAR(10000),
	"brand_id" INT,
	"brand_name_y" VARCHAR(10000),
	"loves_count" INT,
	"rating_y" FLOAT,
	"reviews" FLOAT,
	"size" VARCHAR(10000),
	"variation_type" VARCHAR(10000),
	"variation_value" VARCHAR(10000),
	"ingredients" VARCHAR(10000),
	"price_usd_y" FLOAT,
	"limited_edition" INT,
	"new" INT,
	"online_only" INT,
	"out_of_stock" INT,
	"sephora_exclusive" INT,
	"highlights" VARCHAR(10000),
	"primary_category" VARCHAR(10000),
	"secondary_category" VARCHAR(10000),
	"tertiary_category" VARCHAR(10000),
	"child_count" INT,
	"child_max_price" FLOAT,
	"child_min_price" FLOAT
);

-- Insert data to table 
COPY table_final ("author_id", "rating_x", "is_recommended", "helpfulness",
       "total_feedback_count", "total_neg_feedback_count",
       "total_pos_feedback_count", "submission_time", "review_text",
       "review_title", "skin_tone", "eye_color", "skin_type", "hair_color",
       "product_id", "product_name_x", "brand_name_x", "price_usd_x",
       "product_name_y", "brand_id", "brand_name_y", "loves_count", "rating_y",
       "reviews", "size", "variation_type", "variation_value", "ingredients",
       "price_usd_y", "limited_edition", "new", "online_only", "out_of_stock",
       "sephora_exclusive", "highlights", "primary_category",
       "secondary_category", "tertiary_category", "child_count",
       "child_max_price", "child_min_price") 
FROM 'C:/Users/angel/sephora/celinedion.csv' 
DELIMITER ','
CSV HEADER;

-- Check the table
SELECT *
FROM table_final;