# Databricks notebook source
from pyspark.sql.session import SparkSession
import os

# COMMAND ----------

"""
Below code from https://github.com/fabfitfun/arrietty/blob/master/arrietty/util.py
"""
def get_campaign(survey_id):
    """Get the set of choices for customize, based on campaign id
    Parameters
    ----------
    survey_id: the survey_id of the customize campaign. id in shop_survey_options,
        survey_id in shop_customize_campaigns
    """
    sql = """
        select question_id
            , option_id as field_id
            , name as value
            , sku
        from dw.shop_survey_options
        where id = {}
        order by 1, 2
    """
    sql = sql.format(survey_id)
    campaign = spark_read_sql(sql).toPandas()
    return campaign

def clean_campaign(campaign):
    """Remove surprise me and regional variations of products
    with same name but different SKU."""
    campaign = (campaign.groupby(['value', 'question_id'], as_index=False)
                .first()
                .sort_values(['question_id', 'field_id']))
    campaign = campaign[~(campaign.sku.str.contains('(?i)surprise')) &
                        # ~(campaign.value.str.contains('(?i)Mystery')) &
                        (campaign.value.str.lower() != 'choose for me')].copy()
    campaign['nth_alt'] = (campaign.groupby(['question_id'])
                           .field_id
                           .rank(method='first')
                           - 1)
    return campaign

# COMMAND ----------


