from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window

spark = SparkSession.builder.appName("part3-df").getOrCreate()

first_dataset = spark.read.csv("hdfs://okeanos-master:54310/user/input/Crime_Data_from_2010_to_2019_20231224.csv", header=True, inferSchema=True)
second_dataset = spark.read.csv("hdfs://okeanos-master:54310/user/input/Crime_Data_from_2020_to_Present_20231224.csv", header=True, inferSchema=True)

dataset_df = first_dataset.union(second_dataset)

dataset_df = dataset_df.withColumn("Date Rptd", F.to_timestamp("Date Rptd", "MM/dd/yyyy hh:mm:ss a").cast("date"))
dataset_df = dataset_df.withColumn("DATE OCC", F.to_timestamp("DATE OCC", "MM/dd/yyyy hh:mm:ss a").cast("date"))

dataset_filt_df = (
    dataset_df
    .select(["DATE OCC"])
)

df_top_months = (
    dataset_filt_df
    .withColumn("year", F.year("DATE OCC"))
    .withColumn("month", F.month("DATE OCC"))
    .groupBy("year", "month")
    .agg(F.count("*").alias("crime_total"))
    .withColumn("rank", F.row_number().over(Window.partitionBy("year").orderBy(F.desc("crime_total"))))
    .filter("rank <= 3")
    .orderBy("year", F.desc("crime_total"))
    .withColumnRenamed("rank", "#")
)

df_top_months.show(42)

spark.stop()