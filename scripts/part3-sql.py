from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("part3-sql").getOrCreate()

first_dataset = spark.read.csv(
    "hdfs://okeanos-master:54310/user/input/Crime_Data_from_2010_to_2019_20231224.csv", header=True, inferSchema=True)
second_dataset = spark.read.csv(
    "hdfs://okeanos-master:54310/user/input/Crime_Data_from_2020_to_Present_20231224.csv", header=True, inferSchema=True)

dataset_df = first_dataset.union(second_dataset)

dataset_df = dataset_df.withColumn("Date Rptd", F.to_timestamp(
    "Date Rptd", "MM/dd/yyyy hh:mm:ss a").cast("date"))
dataset_df = dataset_df.withColumn("DATE OCC", F.to_timestamp(
    "DATE OCC", "MM/dd/yyyy hh:mm:ss a").cast("date"))

dataset_df.createOrReplaceTempView("crimes")

filter_query = """
    SELECT
        `DATE OCC`
    FROM
        crimes
"""

dataset_filt_df = spark.sql(filter_query)
dataset_filt_df.createOrReplaceTempView("crimes")

sql_query = """
    WITH ranked_crimes AS (
        SELECT
            YEAR(`DATE OCC`) AS year,
            MONTH(`DATE OCC`) AS month,
            COUNT(*) AS crime_total,
            ROW_NUMBER() OVER (PARTITION BY YEAR(`DATE OCC`) ORDER BY COUNT(*) DESC) AS rank
        FROM crimes
        GROUP BY YEAR(`DATE OCC`), MONTH(`DATE OCC`)
    )
    SELECT
        year,
        month,
        crime_total,
        rank
    FROM
        ranked_crimes
    WHERE
        rank <= 3
    ORDER BY
        year, rank ASC
"""

df_top_months = spark.sql(sql_query)

df_top_months.show(42)

spark.stop()
