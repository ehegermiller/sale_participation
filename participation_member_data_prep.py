# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import Window

# COMMAND ----------

env = 'dev'

# COMMAND ----------

# MAGIC %md 
# MAGIC # Data Prep

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Select columns from User State to keep

# COMMAND ----------

user_state_df = spark.read \
  .format('delta') \
  .load('s3://s3-datascience-prod/user_state/user_state') 

# COMMAND ----------

user_state_df = user_state_df[[
  'account_code'
  , 'date'
  , 'plan_code'
  , 'order_season'
  , 'future_expires_at_date'
  , 'uc_age'
  , 'uc_zipcode_median_income'
  , 'uc_user_region'
  , 'nps_base'
  , 'acq_timestamp'
  , 'subscriber_days_prior_cume'
  , 'acq_discount_cohort'
  , 'acq_box_type'
  , 'acq_type'
  , 'boost_non_sub_participation_ind'
  , 'refills_non_sub_participation_ind'
  , 'choice_non_sub_participation_ind'
  , 'addon_non_sub_participation_ind'
  , 'edit_non_sub_participation_ind'
  , 'promo_non_sub_participation_ind'
  , 'shop_non_sub_participation_ind'
  , 'customize_participation_ind'
  , 'non_sub_cash_collected_amount'
  , 'non_sub_invoice_number_count'
  , 'addon_non_sub_discount_amount'
  , 'boost_non_sub_discount_amount'
  , 'choice_non_sub_discount_amount'
  , 'edit_non_sub_discount_amount'
  , 'promo_non_sub_discount_amount'
  , 'refills_non_sub_discount_amount'
  , 'shop_non_sub_discount_amount'
  , 'addon_non_sub_shipping_amount'
  , 'boost_non_sub_shipping_amount'
  , 'choice_non_sub_shipping_amount'
  , 'edit_non_sub_shipping_amount'
  , 'promo_non_sub_shipping_amount'
  , 'refills_non_sub_shipping_amount'
  , 'shop_non_sub_shipping_amount'
  , 'addon_non_sub_coupon_code'
  , 'boost_non_sub_coupon_code'
  , 'choice_non_sub_coupon_code'
  , 'edit_non_sub_coupon_code'
  , 'promo_non_sub_coupon_code'
  , 'refills_non_sub_coupon_code'
  , 'shop_non_sub_coupon_code'
  , 'csat_satisfied_ind'
  , 'subscriber_days_prior_cume'
]]

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Exclude trial members, employee accounts and test accounts

# COMMAND ----------

user_state_non_trial = user_state_df \
  .filter(col('plan_code').isin('VIP', 'VIPA')) \
  .filter(~col('account_code').contains("fff.com")) \
  .filter(~col('account_code').contains("%fabfitfun.com%")) 

# COMMAND ----------

fpa_test_accounts = spark.sql("""SELECT distinct account_code FROM fpa_prod.fpa_test_accounts""")

fpa_test_accounts = fpa_test_accounts \
.withColumnRenamed('account_code', 'test_account_code')

if env == 'dev':
  print(fpa_test_accounts.count())
else: 
  None

# COMMAND ----------

df = user_state_non_trial \
.join(fpa_test_accounts, user_state_non_trial.account_code == fpa_test_accounts.test_account_code, how='left') \
.filter(col('test_account_code').isNull()) \
.drop(col('test_account_code'))

if env == 'dev':
  print(df.count())
else: 
  None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add calculated columns

# COMMAND ----------

user_lifetime_current_day_window = Window.partitionBy('account_code').orderBy('date', 'plan_code')
season_window = Window.partitionBy('account_code').orderBy('date', 'plan_code', 'order_season')
days = lambda i: i * 86400 
in_last_100_days_window = Window.partitionBy('account_code').orderBy(col('date').cast('timestamp').cast('long')).rangeBetween(days(-101), days(-1))

df = df \
.withColumn('acq_date', to_date('acq_timestamp')) \
.withColumn('months_since_acq', months_between(col('date'), col('acq_date'))) \
.withColumn("agg_participation_ind", when(df.boost_non_sub_participation_ind == 1, 1) \
  .when(df.refills_non_sub_participation_ind == 1, 1)
  .when(df.choice_non_sub_participation_ind == 1, 1)
  .when(df.addon_non_sub_participation_ind == 1, 1)
  .when(df.edit_non_sub_participation_ind == 1, 1)
  .when(df.promo_non_sub_participation_ind == 1, 1)
  .when(df.shop_non_sub_participation_ind == 1, 1)
  .otherwise(0)) \
.withColumn("participation_count", col("boost_non_sub_participation_ind") \
                   + col("refills_non_sub_participation_ind") \
                   + col("choice_non_sub_participation_ind") \
                   + col("addon_non_sub_participation_ind") \
                   + col("edit_non_sub_participation_ind") \
                   + col("promo_non_sub_participation_ind") \
                   + col("shop_non_sub_participation_ind")) \
.withColumn('days_to_expiration', datediff(col('future_expires_at_date'), col('date'))) \
.withColumn('expires_within_90_days_ind', when(col('days_to_expiration') <= 90, 1).otherwise(0)) \
.withColumn('day_of_last_participation', when(col('agg_participation_ind') == 1, col('date'))) \
.withColumn('days_since_last_participation', last(col('day_of_last_participation'), ignorenulls=True).over(user_lifetime_current_day_window)) \
.withColumn('past_customize_participant', sum('customize_participation_ind').over(user_lifetime_current_day_window)) \
.withColumn('past_customize_participant', when(col('past_customize_participant') >= 1, 1).otherwise(0)) \
.withColumn('past_ecomm_participant', sum('agg_participation_ind').over(user_lifetime_current_day_window)) \
.withColumn('past_ecomm_participant', when(col('past_ecomm_participant') >= 1, 1).otherwise(0)) \
.withColumn('season_to_date_participation', sum('agg_participation_ind').over(season_window)) \
.withColumn('seasonality', col('order_season').cast(StringType())) \
.withColumn('seasonality', col('seasonality')[3:2]) \
.withColumn('participation_in_last_100_days', sum('agg_participation_ind').over(in_last_100_days_window))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add revenue columns

# COMMAND ----------

df = df \
.withColumn('total_discount_amt', col('addon_non_sub_discount_amount') 
            + col('boost_non_sub_discount_amount') 
            + col('choice_non_sub_discount_amount') 
            + col('edit_non_sub_discount_amount') 
            + col('promo_non_sub_discount_amount') 
            + col('refills_non_sub_discount_amount') 
            + col('shop_non_sub_discount_amount')) \
.withColumn('total_shipping_amt', col('addon_non_sub_shipping_amount') 
            + col('boost_non_sub_shipping_amount') 
            + col('choice_non_sub_shipping_amount') 
            + col('edit_non_sub_shipping_amount') 
            + col('promo_non_sub_shipping_amount') 
            + col('refills_non_sub_shipping_amount')
            + col('shop_non_sub_shipping_amount')) \
.withColumn('total_order_revenue', col('non_sub_cash_collected_amount') - col('total_discount_amt') + col('total_shipping_amt')) \
.withColumn('avg_order_revenue', col('total_order_revenue') / col('non_sub_invoice_number_count')) \
.withColumn('last_order_revenue', last(col('total_order_revenue'), ignorenulls=True).over(user_lifetime_current_day_window)) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Drop columns no longer needed

# COMMAND ----------

df = df.drop('acq_date'
             , 'acq_ind'
             , 'acq_timestamp'
             , 'future_expires_at_date'
             , 'days_to_expiration'
             , 'total_discount_amt'
             , 'total_shipping_amt'
             , 'addon_non_sub_discount_amount' 
             , 'boost_non_sub_discount_amount'
             , 'choice_non_sub_discount_amount'
             , 'edit_non_sub_discount_amount'
             , 'promo_non_sub_discount_amount'
             , 'refills_non_sub_discount_amount'
             , 'shop_non_sub_discount_amount'
             , 'addon_non_sub_shipping_amount' 
             , 'boost_non_sub_shipping_amount'
             , 'choice_non_sub_shipping_amount'
             , 'edit_non_sub_shipping_amount' 
             , 'promo_non_sub_shipping_amount'
             , 'refills_non_sub_shipping_amount'
             , 'shop_non_sub_shipping_amount'
            )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add participation indicators within 30, 60, 90, 120 days

# COMMAND ----------

participate_within_30_days_window = Window.partitionBy('account_code').orderBy(col('date').cast('timestamp').cast('long')).rangeBetween(days(+1), days(+31))
participate_within_60_days_window = Window.partitionBy('account_code').orderBy(col('date').cast('timestamp').cast('long')).rangeBetween(days(+1), days(+61))
participate_within_90_days_window = Window.partitionBy('account_code').orderBy(col('date').cast('timestamp').cast('long')).rangeBetween(days(+1), days(+91))
participate_within_120_days_window = Window.partitionBy('account_code').orderBy(col('date').cast('timestamp').cast('long')).rangeBetween(days(+1), days(+121))

df = df \
.withColumn('participates_within_30_days', sum('agg_participation_ind').over(participate_within_30_days_window)) \
.withColumn('participates_within_60_days', sum('agg_participation_ind').over(participate_within_60_days_window)) \
.withColumn('participates_within_90_days', sum('agg_participation_ind').over(participate_within_90_days_window)) \
.withColumn('participates_within_120_days', sum('agg_participation_ind').over(participate_within_120_days_window)) \
.withColumn('participates_within_30_days_ind', when(col('participates_within_30_days') >= 1, 1).otherwise(0)) \
.withColumn('participates_within_60_days_ind', when(col('participates_within_60_days') >= 1, 1).otherwise(0)) \
.withColumn('participates_within_90_days_ind', when(col('participates_within_90_days') >= 1, 1).otherwise(0)) \
.withColumn('participates_within_120_days_ind', when(col('participates_within_120_days') >= 1, 1).otherwise(0)) 

if env == 'dev':
  display(df.limit(10))
else: 
  None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exclude seasons prior to 1901

# COMMAND ----------

df = df \
.filter(col('order_season') >= '1901')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean up columns

# COMMAND ----------

df = df \
.withColumn('uc_age', when(col('uc_age').isNull(), 'N/A').when((col('uc_age') < 10) | (col('uc_age') > 80), 'Other').otherwise(col('uc_age'))) \
.withColumn('uc_zipcode_median_income', when(col('uc_zipcode_median_income').isNull(), 'N/A').otherwise(col('uc_zipcode_median_income'))) \
.withColumn('uc_user_region', when(col('uc_user_region').isNull(), 'N/A').otherwise(col('uc_user_region'))) \
.withColumn('nps_base', when(col('nps_base').isNull(), 'N/A').otherwise(col('nps_base'))) \
.withColumn('acq_discount_cohort', when(col('acq_discount_cohort').isNull(), 'N/A').otherwise(col('acq_discount_cohort'))) \
.withColumn('acq_box_type', when(col('acq_box_type').isNull(), 'N/A').otherwise(col('acq_box_type'))) \
.withColumn('acq_type', when(col('acq_type').isNull(), 'N/A').otherwise(col('acq_type'))) \
.withColumn('addon_non_sub_coupon_code', when(col('addon_non_sub_coupon_code').isNull(), 'N/A').when(col('addon_non_sub_coupon_code') == '', 'Blank').otherwise(col('addon_non_sub_coupon_code'))) \
.withColumn('boost_non_sub_coupon_code', when(col('boost_non_sub_coupon_code').isNull(), 'N/A').when(col('boost_non_sub_coupon_code') == '', 'Blank').otherwise(col('boost_non_sub_coupon_code'))) \
.withColumn('choice_non_sub_coupon_code', when(col('choice_non_sub_coupon_code').isNull(), 'N/A').when(col('choice_non_sub_coupon_code') == '', 'Blank').otherwise(col('choice_non_sub_coupon_code'))) \
.withColumn('edit_non_sub_coupon_code', when(col('edit_non_sub_coupon_code').isNull(), 'N/A').when(col('edit_non_sub_coupon_code') == '', 'Blank').otherwise(col('edit_non_sub_coupon_code'))) \
.withColumn('promo_non_sub_coupon_code', when(col('promo_non_sub_coupon_code').isNull(), 'N/A').when(col('promo_non_sub_coupon_code') == '', 'Blank').otherwise(col('promo_non_sub_coupon_code'))) \
.withColumn('refills_non_sub_coupon_code', when(col('refills_non_sub_coupon_code').isNull(), 'N/A').when(col('refills_non_sub_coupon_code') == '', 'Blank').otherwise(col('refills_non_sub_coupon_code'))) \
.withColumn('shop_non_sub_coupon_code', when(col('shop_non_sub_coupon_code').isNull(), 'N/A').when(col('shop_non_sub_coupon_code') == '', 'Blank').otherwise(col('shop_non_sub_coupon_code'))) \
.withColumn('csat_satisfied_ind', when(col('csat_satisfied_ind').isNull(), 'N/A').otherwise(col('csat_satisfied_ind'))) \
.withColumn('months_since_acq', round(col('months_since_acq'), 2)) \
.withColumn('months_since_acq', when(col('months_since_acq').isNull(), 'N/A').otherwise(col('months_since_acq'))) \
.withColumn('day_of_last_participation', when(col('day_of_last_participation').isNull(), 'N/A').otherwise(col('day_of_last_participation'))) \
.withColumn('days_since_last_participation', when(col('days_since_last_participation').isNull(), 'N/A').otherwise(col('days_since_last_participation'))) \
.withColumn('participation_in_last_100_days', when(col('participation_in_last_100_days').isNull(), 'N/A').otherwise(col('participation_in_last_100_days'))) \
.withColumn('total_order_revenue', when(col('total_order_revenue').isNull(), 'N/A').otherwise(col('total_order_revenue'))) \
.withColumn('avg_order_revenue', when(col('avg_order_revenue').isNull(), 'N/A').otherwise(col('avg_order_revenue'))) \
.withColumn('last_order_revenue', when(col('last_order_revenue').isNull(), 'N/A').otherwise(col('last_order_revenue'))) \
.withColumn('participates_within_30_days', when(col('participates_within_30_days').isNull(), 0).otherwise(col('participates_within_30_days'))) \
.withColumn('participates_within_60_days', when(col('participates_within_60_days').isNull(), 0).otherwise(col('participates_within_60_days'))) \
.withColumn('participates_within_90_days', when(col('participates_within_90_days').isNull(), 0).otherwise(col('participates_within_90_days'))) \
.withColumn('participates_within_120_days', when(col('participates_within_120_days').isNull(), 0).otherwise(col('participates_within_120_days'))) 

# COMMAND ----------

# MAGIC %md 
# MAGIC ## QA columns to confirm there are no nulls

# COMMAND ----------

df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### Age

# COMMAND ----------

age = df.groupBy('uc_age').agg(countDistinct(col('account_code')).alias('count'))
display(age.orderBy('uc_age'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Plan Code

# COMMAND ----------

plan_code = df.groupBy('plan_code').agg(countDistinct(col('account_code')).alias('count'))
display(plan_code.orderBy('plan_code'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Order Season

# COMMAND ----------

df.select('order_season').distinct().show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Income

# COMMAND ----------

zipcode = df.groupBy('uc_zipcode_median_income').agg(countDistinct(col('account_code')).alias('count'))
display(zipcode.orderBy('uc_zipcode_median_income'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### User Region

# COMMAND ----------

regions = df.groupBy('uc_user_region').agg(countDistinct(col('account_code')).alias('count'))
display(regions.orderBy('uc_user_region'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### NPS Base

# COMMAND ----------

nps_base = df.groupBy('nps_base').agg(countDistinct(col('account_code')).alias('count'))
display(nps_base.orderBy('nps_base'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### subscriber_days_prior_cume

# COMMAND ----------

sub_days = df.groupBy('subscriber_days_prior_cume').agg(countDistinct(col('account_code')).alias('count'))
display(sub_days.orderBy('subscriber_days_prior_cume'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### acq_discount_cohort

# COMMAND ----------

acq_disc = df.groupBy('acq_discount_cohort').agg(countDistinct(col('account_code')).alias('count'))
display(acq_disc.orderBy('acq_discount_cohort'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### acq_box_type

# COMMAND ----------

acq = df.groupBy('acq_box_type').agg(countDistinct(col('account_code')).alias('count'))
display(acq.orderBy('acq_box_type'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### acq_type

# COMMAND ----------

acq_type = df.groupBy('acq_type').agg(countDistinct(col('account_code')).alias('count'))
display(acq_type.orderBy('acq_type'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### addon_non_sub_coupon_code

# COMMAND ----------

addon = df.groupBy('addon_non_sub_coupon_code').agg(countDistinct(col('account_code')).alias('count'))
display(addon.orderBy('addon_non_sub_coupon_code'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### boost_non_sub_coupon_code

# COMMAND ----------

boost = df.groupBy('boost_non_sub_coupon_code').agg(countDistinct(col('account_code')).alias('count'))
display(boost.orderBy('boost_non_sub_coupon_code'))

# COMMAND ----------

# MAGIC %md 
# MAGIC #### choice_non_sub_coupon_code

# COMMAND ----------

choice = df.groupBy('choice_non_sub_coupon_code').agg(countDistinct(col('account_code')).alias('count'))
display(choice.orderBy('choice_non_sub_coupon_code'))

# COMMAND ----------

# MAGIC %md 
# MAGIC #### edit_non_sub_coupon_code

# COMMAND ----------

edit = df.groupBy('edit_non_sub_coupon_code').agg(countDistinct(col('account_code')).alias('count'))
display(edit.orderBy('edit_non_sub_coupon_code'))

# COMMAND ----------

# MAGIC %md 
# MAGIC #### promo_non_sub_coupon_code

# COMMAND ----------

promo = df.groupBy('promo_non_sub_coupon_code').agg(countDistinct(col('account_code')).alias('count'))
display(promo.orderBy('promo_non_sub_coupon_code'))

# COMMAND ----------

# MAGIC %md 
# MAGIC #### refills_non_sub_coupon_code

# COMMAND ----------

refills = df.groupBy('refills_non_sub_coupon_code').agg(countDistinct(col('account_code')).alias('count'))
display(refills.orderBy('refills_non_sub_coupon_code'))

# COMMAND ----------

# MAGIC %md 
# MAGIC #### shop_non_sub_coupon_code

# COMMAND ----------

shop = df.groupBy('shop_non_sub_coupon_code').agg(countDistinct(col('account_code')).alias('count'))
display(shop.orderBy('shop_non_sub_coupon_code'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### csat_satisfied_ind

# COMMAND ----------

csat = df.groupBy('csat_satisfied_ind').agg(countDistinct(col('account_code')).alias('count'))
display(csat.orderBy('csat_satisfied_ind'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### months_since_acq

# COMMAND ----------

months_since_acq = df.groupBy('months_since_acq').agg(countDistinct(col('account_code')).alias('count'))
display(months_since_acq.orderBy('months_since_acq'))

# COMMAND ----------

# MAGIC %md 
# MAGIC #### agg_participation_ind

# COMMAND ----------

df.select('agg_participation_ind').distinct().show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### participation_count

# COMMAND ----------

df.select('participation_count').distinct().show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### expires_within_90_days_ind

# COMMAND ----------

df.select('expires_within_90_days_ind').distinct().show()

# COMMAND ----------

# MAGIC %md 
# MAGIC #### day_of_last_participation

# COMMAND ----------

df.select('day_of_last_participation').distinct().show()

# COMMAND ----------

# MAGIC %md 
# MAGIC #### days_since_last_participation

# COMMAND ----------

df.select('days_since_last_participation').distinct().show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### past_customize_participant

# COMMAND ----------

df.select('past_customize_participant').distinct().show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### past_ecomm_participant

# COMMAND ----------

df.select('past_ecomm_participant').distinct().show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### season_to_date_participation

# COMMAND ----------

season_to_date = df.groupBy('season_to_date_participation').agg(countDistinct(col('account_code')).alias('count'))
display(season_to_date.orderBy('season_to_date_participation'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### seasonality

# COMMAND ----------

df.select('seasonality').distinct().show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### participation_in_last_100_days

# COMMAND ----------

df.select('participation_in_last_100_days').distinct().show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### total_order_revenue

# COMMAND ----------

df.select('total_order_revenue').distinct().show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### avg_order_revenue

# COMMAND ----------

df.select('avg_order_revenue').distinct().show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### last_order_revenue

# COMMAND ----------

df.select('last_order_revenue').distinct().show()

# COMMAND ----------

# MAGIC %md 
# MAGIC #### participates_within_30_days

# COMMAND ----------

df.select('participates_within_30_days').distinct().show()

# COMMAND ----------

# MAGIC %md 
# MAGIC #### participates_within_30_days_ind

# COMMAND ----------

df.select('participates_within_30_days_ind').distinct().show()

# COMMAND ----------

# MAGIC %md 
# MAGIC #### subscriber_days_prior_cume

# COMMAND ----------

df.select('subscriber_days_prior_cume').distinct().orderBy('subscriber_days_prior_cume').show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check for NULLs

# COMMAND ----------

# Columns with empty values and NULLs
from pyspark.sql.functions import isnan,when,count

columns_with_nulls = df.agg(*[count(when(col(c).contains('None') | \
                            col(c).contains('NULL') | \
                            (col(c) == '' ) | \
                            col(c).isNull(), c 
                           )).alias(c)
                    for c in df.columns])

display(columns_with_nulls)

# COMMAND ----------


