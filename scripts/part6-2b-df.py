from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, StructType, StructField, StringType

spark = SparkSession.builder.appName("part6-2b-df").getOrCreate()

spark.sparkContext.addPyFile("/home/user/scripts/constants.py")
spark.sparkContext.addPyFile("/home/user/scripts/geodesiccapability.py")
spark.sparkContext.addPyFile("/home/user/scripts/geomath.py")
spark.sparkContext.addPyFile("/home/user/scripts/geodesic.py")
spark.sparkContext.addPyFile("/home/user/scripts/units.py")
spark.sparkContext.addPyFile("/home/user/scripts/util.py")
spark.sparkContext.addPyFile("/home/user/scripts/format.py")
spark.sparkContext.addPyFile("/home/user/scripts/point.py")
spark.sparkContext.addPyFile("/home/user/scripts/distance.py")

import distance


first_dataset = spark.read.csv("hdfs://okeanos-master:54310/user/input/Crime_Data_from_2010_to_2019_20231224.csv", header=True, inferSchema=True)
second_dataset = spark.read.csv("hdfs://okeanos-master:54310/user/input/Crime_Data_from_2020_to_Present_20231224.csv", header=True, inferSchema=True)

dataset_df = first_dataset.union(second_dataset)

dataset_df = dataset_df.withColumn("Date Rptd", F.to_timestamp("Date Rptd", "MM/dd/yyyy hh:mm:ss a").cast("date"))
dataset_df = dataset_df.withColumn("DATE OCC", F.to_timestamp("DATE OCC", "MM/dd/yyyy hh:mm:ss a").cast("date"))

policedept_df = spark.read.csv("hdfs://okeanos-master:54310/user/input/LAPD_Police_Stations.csv", header=True, inferSchema=True)

dataset_filt_df = (
    dataset_df
    .filter(F.col("LAT") != 0.0)
    .filter(F.col("Weapon Used Cd").isNotNull())
    .select(["DR_NO", "LAT", "LON"])
)

policedept_filt_df = (
    policedept_df
    .select(["DIVISION", "X", "Y"])
    .withColumnRenamed("DIVISION", "division")
)

def dist(x1, y1, x2, y2):
    return distance.geodesic((x1, y1), (x2, y2)).km

dist_udf = F.udf(dist, FloatType())

divisions = [row.division for row in policedept_filt_df.select("division").distinct().collect()]
division_data = policedept_filt_df.filter(F.col("division") == divisions[0])
joined_df = dataset_filt_df.crossJoin(division_data)
dataset_filt_df = joined_df.withColumn("Distance", dist_udf(F.col("Y"), F.col("X"), F.col("LAT"), F.col("LON")))
dataset_filt_df = dataset_filt_df.drop("X", "Y")
dataset_filt_df = dataset_filt_df.withColumnRenamed("division", "closest_division")

# Iterate through each division
for division in divisions[1:]:
    division_data = policedept_filt_df.filter(F.col("division") == division)
    joined_df = dataset_filt_df.crossJoin(division_data)
    joined_df = joined_df.withColumn("New_Distance", dist_udf(F.col("Y"), F.col("X"), F.col("LAT"), F.col("LON")))
    joined_df = joined_df.withColumn("Distance", F.least(F.col("Distance"), F.col("New_Distance")))
    joined_df = joined_df.withColumn("distance_equal", F.col("Distance") == F.col("New_Distance"))
    joined_df = joined_df.withColumn("closest_division", F.when(F.col("distance_equal"), F.col("division")).otherwise(F.col("closest_division")))
    dataset_filt_df = joined_df.drop("New_Distance", "distance_equal", "X", "Y", "division")

final_df = (
    dataset_filt_df
    .groupBy("closest_division")
    .agg(F.avg("Distance").alias("average_distance"), 
        F.count("closest_division").alias("#")
    )
    .orderBy(F.col("#").desc())
)

final_df.show(21)

spark.stop()
