from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

spark = SparkSession.builder.appName("part4-df").getOrCreate()

first_dataset = spark.read.csv(
    "hdfs://okeanos-master:54310/user/input/Crime_Data_from_2010_to_2019_20231224.csv", header=True, inferSchema=True)
second_dataset = spark.read.csv(
    "hdfs://okeanos-master:54310/user/input/Crime_Data_from_2020_to_Present_20231224.csv", header=True, inferSchema=True)

dataset_df = first_dataset.union(second_dataset)

dataset_df = dataset_df.withColumn("Date Rptd", F.to_timestamp(
    "Date Rptd", "MM/dd/yyyy hh:mm:ss a").cast("date"))
dataset_df = dataset_df.withColumn("DATE OCC", F.to_timestamp(
    "DATE OCC", "MM/dd/yyyy hh:mm:ss a").cast("date"))

dataset_filt_df = (
    dataset_df
    .select(["TIME OCC", "Premis Cd"])
)


def calc_part(time):
    if time >= 500 and time <= 1159:
        return "morning"
    if time >= 1200 and time <= 1659:
        return "afternoon"
    if time >= 1700 and time <= 2059:
        return "evening"
    return "night"


calc_part_udf = F.udf(calc_part, StringType())

df_timeofday = (
    dataset_filt_df
    .withColumn("partofday", calc_part_udf(F.col("TIME OCC")))
    .filter(F.col("Premis Cd") == 101)
    .groupBy("partofday")
    .agg(F.count("*").alias("crime_total"))
    .orderBy(F.desc("crime_total"))
)

df_timeofday.show(4)

spark.stop()
