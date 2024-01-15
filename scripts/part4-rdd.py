from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Row

spark = SparkSession.builder.appName("part4-rdd").getOrCreate()

first_dataset = spark.read.csv(
    "hdfs://okeanos-master:54310/user/input/Crime_Data_from_2010_to_2019_20231224.csv", header=True, inferSchema=True)
second_dataset = spark.read.csv(
    "hdfs://okeanos-master:54310/user/input/Crime_Data_from_2020_to_Present_20231224.csv", header=True, inferSchema=True)

dataset_df = first_dataset.union(second_dataset)

dataset_df = dataset_df.withColumn("Date Rptd", F.to_timestamp(
    "Date Rptd", "MM/dd/yyyy hh:mm:ss a").cast("date"))
dataset_df = dataset_df.withColumn("DATE OCC", F.to_timestamp(
    "DATE OCC", "MM/dd/yyyy hh:mm:ss a").cast("date"))

dataset_rdd = dataset_df.rdd


def calc_part(row):
    time = row["TIME OCC"]
    if time >= 500 and time <= 1159:
        part_of_day = "morning"
    elif time >= 1200 and time <= 1659:
        part_of_day = "afternoon"
    elif time >= 1700 and time <= 2059:
        part_of_day = "evening"
    else:
        part_of_day = "night"
    return Row(partofday=part_of_day, Premis_Cd=row["Premis Cd"])


rdd_partofday = dataset_rdd.map(calc_part)
rdd_partofday = rdd_partofday.filter(lambda x: x["Premis_Cd"] == 101)
rdd_partofday = rdd_partofday.map(lambda x: (
    x["partofday"], 1)).reduceByKey(lambda x, y: x + y)
rdd_partofday = rdd_partofday.sortBy(lambda x: x[1], ascending=False)

print(rdd_partofday.take(4))
