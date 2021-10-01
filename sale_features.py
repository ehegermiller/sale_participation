# Databricks notebook source
from functools import reduce
import re
import pprint
import boto3
import arrow
from itertools import chain
import pyspark.sql.functions as f
from pyspark.sql import Window
from pyspark.sql.types import StructField, StructType, StringType, IntegerType
import recommendation_util
from recommendation_util import ecomm_util as _, cosine as __, json_util as ___, transformation_util as ____

# COMMAND ----------

# MAGIC %run ./util

# COMMAND ----------

# Determine environment
arn = boto3.client('sts').get_caller_identity()['Arn']
if 'ds_developer_role' in arn:
    env = 'dev'
elif 'ds_prod_role' in arn:
    env = 'prod'
else:
    raise NotImplementedError('Expected ARN to contain ds_developer_role or ds_prod_role')
    
# Swap the secrets used
secret_scope = dict(dev='emma', prod='algorithms-prod')[env]

jdbc_url_string = ''.join([
  "jdbc:redshift://",
  dbutils.secrets.get(scope = secret_scope, key = "redshift_host"), ":",
  dbutils.secrets.get(scope = secret_scope, key = "redshift_port"), "/",
  dbutils.secrets.get(scope = secret_scope, key = "redshift_db"), "?",
  "user=", dbutils.secrets.get(scope = secret_scope, key = "redshift_user"), 
  "&password=", dbutils.secrets.get(scope = secret_scope, key = "redshift_password")
])

# What date are these recommendations build built for?
rec_date = arrow.utcnow().shift(days=-1).format('YYYY-MM-DD') 
assert arrow.get(rec_date) < arrow.utcnow().floor('day'), f"rec_date should be yesterday or before: {rec_date}"

feature_env = 'prod'
product_features_s3 = f's3://s3-datascience-{feature_env}/features/product/'

# COMMAND ----------

# MAGIC %md
# MAGIC # Pull Datasets

# COMMAND ----------

# MAGIC %md
# MAGIC ## Campaigns and Products
# MAGIC Data handling comes from [fivetran_box_rating_box documentation](https://fffptc.atlassian.net/wiki/spaces/ECOMM/pages/641532207/Box+Database)

# COMMAND ----------

# Captures shop, new member, boost, add-on, edit

campaign_product_query = """
SELECT DISTINCT
  c.id as campaign_id
  , c.name
  /* , rms.subchannel */
  , c.type
  , c.start
  , c.start_premium
  , c.stop
  , c.stop_premium
  , v.sku
  , v.product_id
  , v.retail_price
  , cv.sale_price
  , cv.stock_total - cv.stock_out AS available_units
  , cv.is_deleted
  , cv.modified_ts
  /* , COUNT(*) OVER (PARTITION BY c.id, v.sku) AS row_dup_count */
FROM fivetran_box_rating_box.v_campaigns c 
INNER JOIN fivetran_box_rating_box.v_campaign_variants cv ON c.id = cv.campaign_id
INNER JOIN fivetran_box_rating_box.v_variants v ON cv.variant_id = v.id
/* LEFT JOIN analytics_prod.revenue_member_sku rms ON c.id = rms.campaign_id */
WHERE v.sku IS NOT NULL
"""
# Filter out c_v.is_deleted = 1 AND c_v.modified_ts < c.start

# Discount variants.retail_price campaign_variants.sale_price

print(f"Querying Redshift:\n{campaign_product_query}")

campaign_product_df = (
  spark.read
  .format("com.databricks.spark.redshift")
  .option("url", jdbc_url_string)
  .option("query", campaign_product_query)
  .option("tempdir", f's3://s3-datascience-{env}/temp/')
  .option('forward_spark_s3_credentials', True)
  .load()
  .withColumn('start', f.least(f.col('start'), f.col('start_premium')))
  .withColumn('stop', f.greatest(f.col('stop'), f.col('stop_premium')))
  .withColumn('deleted_prior_to_start', 
              f.when((f.col('is_deleted')==1) & (f.col('modified_ts') < f.col('start')), 1).otherwise(0))
  # Remove products that were deleted before the start of the campaign
  .filter(f.col('deleted_prior_to_start')==0) 
  # Remove new member Add On campaigns
  .filter(f.col('type').isin(['EDIT', 'SHOP', 'SEASON', 'BOOST'])) 
  # Remove test and placeholder campaigns
  .filter(~f.col('name').rlike('(?i)test|placeholder')) 
  # Remove rows with no start or stop
  .filter((f.col('start').isNotNull()) & (f.col('stop').isNotNull()))
  .drop('deleted_prior_to_start', 'start_premium', 'stop_premium')
  .cache()
)
display(campaign_product_df)

# Campaign count by type
display(campaign_product_df.select('type', 'campaign_id').distinct().groupby('type').count())

# COMMAND ----------

# MAGIC %md
# MAGIC TO RESOLVE
# MAGIC - `SHOP` has no start/stop
# MAGIC - `revenue_member_sku` pulls `campaign_id` from `dw_dim_campaign_product` and `subchannel` from `dw_dim_sales_channel` (`item_type` from `dw_dim_invoice_line_items`)

# COMMAND ----------

# Check for duplicated rows for subset campaign_id and sku
dup_campaign_product_df = (
  campaign_product_df
  .groupby('campaign_id', 'sku')
  .count()
  .filter(f.col('count')>1)
  .cache()
  .join(campaign_product_df, on=['campaign_id', 'sku'], how='left')
  .orderBy('campaign_id', 'sku')
)

# Check for products that where deleted before campaign start
deleted_campaign_product_df = (
  campaign_product_df
  .filter((f.col('is_deleted')==1) & (f.col('modified_ts') < f.col('start')))
)

assert dup_campaign_product_df.count() == 0
assert deleted_campaign_product_df.count() == 0

print("Count of null cells by column")
display(campaign_product_df.select([f.count(f.when(f.col(c).isNull(), c)).alias(c) for c in campaign_product_df.columns]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional Choice

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## reFill

# COMMAND ----------

# MAGIC %md
# MAGIC ## Product features

# COMMAND ----------

product_feature_df = (
  recommendation_util.common.pull_delta_data_partitioned_by_date(product_features_s3, rec_date)
  .filter((f.col("timezone") == 'utc') & (f.col('version') == "20200801"))
  .distinct()  # Workaround for bug NLTCS-757
  # Just select the needed features
  .transform(
    lambda mdf: mdf.select(
      'sku', 
      'product_category_value',
      'product_subcategory1_value',
      *[acol for acol in mdf.columns if re.compile('^product_doc2vec_index_.*_value$').match(acol)]
  ))
  .dropDuplicates()
  .cache()
)

display(product_feature_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Campaigns and Products

# COMMAND ----------

# ONLY HAS OVERLAP FOR campaign_id FOR subchannels EDIT, SEASON and NEW MEMBER 

# # Pull campaign and product information from Redshift
# campaign_product_query = """
# SELECT DISTINCT 
#   LOWER(dsc.subchannel) as subchannel,
#   dcp.campaign_type,
#   dcp.campaign_id,
#   dcp.campaign_title, 
#   MAX(dcp.total_stock) OVER (PARTITION BY dcp.campaign_id, dcp.sku) AS total_stock,
#   dcp.product_id,
#   dcp.sku,
#   dcp.msrp,
#   dcp.fff_sale_price,
#   dcp.update_date as update_date_pst_dcp,
#   c.name,
#   c.type,
#   c.start_premium,
#   c.start,
#   c.stop_premium,
#   c.stop
# FROM fivetran_box_rating_box.v_campaigns AS c
# INNER JOIN public.dw_dim_campaign_product AS dcp ON c.id = dcp.campaign_id
# INNER JOIN  public.dw_fact_line_item AS fli ON fli.campaign_product_id = dcp.campaign_product_id
# INNER JOIN public.dw_dim_sales_channel AS dsc ON  dsc.sales_channel_id = fli.sales_channel_id
# """
# # WHERE c.id IS NOT NULL
# # AND dcp.sku IS NOT NULL


# print(f"Querying Redshift:\n{campaign_product_query}")

# w = Window.partitionBy('campaign_id', 'sku')
# campaign_product_df = (
#   spark.read
#   .format("com.databricks.spark.redshift")
#   .option("url", jdbc_url_string)
#   .option("query", campaign_product_query)
#   .option("tempdir", f's3://s3-datascience-{env}/temp/')
#   .option('forward_spark_s3_credentials', True)
#   .load()
#   .withColumn('start', f.least('start', 'start_premium'))
#   .withColumn('stop', f.greatest('stop', 'stop_premium'))
#   .withColumn('subchannel_dup', f.when((f.count('subchannel').over(w) >1) & (f.col('subchannel')!=f.col('campaign_type')), 1).otherwise(0))
#   .filter(f.col('subchannel_dup')==0)
#   .drop('subchannel_dup', 'start_premium', 'stop_premium')
#   .cache()
# )

# COMMAND ----------

# MAGIC %md
# MAGIC # Calculate Features

# COMMAND ----------

# mapping = {
#   1 : 'spring',
#   2 : 'summer',
#   3 : 'fall',
#   4 : 'winter'
# }
# mapping_expr = f.create_map([f.lit(x) for x in chain(*mapping.items())])

# .withColumn('season', mapping_expr.getItem(f.col('order_season').substr(-1,1)))

# COMMAND ----------

# MAGIC %md
# MAGIC - year
# MAGIC - sale duration in days
# MAGIC 
# MAGIC 
# MAGIC - number of products in sale
# MAGIC - inventory available
# MAGIC - average units/product_id
# MAGIC - median mrsp
# MAGIC - average mrsp
# MAGIC - product variance
# MAGIC - percent of product_id's seen before
# MAGIC - average product level discount
# MAGIC - median product level discount

# COMMAND ----------

display(campaign_product_df.select('campaign_id', 'name', 'start').distinct().orderBy('start'))


# COMMAND ----------

previously_seen_w = Window.partitionBy('sku').orderBy(f.col('start'))

sale_features = (
  campaign_product_df
  .join(product_feature_df, on='sku', how='left')
  .withColumn('year', f.year(f.to_date(f.col('start'))))
  .withColumn('duration', f.datediff(f.to_date(f.col('stop')), f.to_date(f.col('start'))))
#   .withColumn('percent_discount', 
#               f.round((f.col('retail_price')-f.col('sale_price'))/f.col('retail_price'), 2))
#   .withColumn('previously_sold', 
#               f.when(f.lag('start', 1).over(previously_seen_w).isNotNull(), 1).otherwise(0))
  
  
#   .
#   .withColumn('lag_start', f.lag('start', 1).over(w))
  .cache()
)
# display(sale_features.select([f.count(f.when(f.col(c).isNull(), c)).alias(c) for c in sale_features.columns]))
# display(sale_features.select('sku', 'campaign_id', 'name', 'start', 'previously_sold').orderBy('product_id', 'campaign_id'))

# COMMAND ----------

# MAGIC %md 
# MAGIC # Exploratory Data Analysis

# COMMAND ----------

# Summary of null values by column
print("Campaign count of null cells by column")
display(campaign_df.select([f.count(f.when(f.col(c).isNull(), c)).alias(c) for c in campaign_df.columns]))

print("Variant count of null cells by column")
display(variant_df.select([f.count(f.when(f.col(c).isNull(), c)).alias(c) for c in variant_df.columns]))

print("Product feature count of null cells by column")
display(product_feature_df.select([f.count(f.when(f.col(c).isNull(), c)).alias(c) for c in product_feature_df.columns]))

# COMMAND ----------

# variant_df.select('sale_price', 'retail_price', 'stock_total').summary().show()
display(campaign_product_df.groupby('type').count())

# COMMAND ----------

# MAGIC %md
# MAGIC # Cleaning

# COMMAND ----------

# campaign name with Beta test?? or any Test or OLD
# How does the campaign change over the 30 days spands
# New member
# Check that products classified in subchannel are in campaign type -- analytics_prod.revenue_member_sku campaign_id, subchannel, sku in v_campaign and v_variants
#janky check on which products are available one month out using cv.created_ts and cv.is_deleted

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Manipulations
# MAGIC ## Imputation
# MAGIC ## Categorical to Number
# MAGIC ## Normalized
