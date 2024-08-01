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