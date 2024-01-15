import distance
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType

spark = SparkSession.builder.appName("part6-df-1b-merge").getOrCreate()

spark.sparkContext.addPyFile("/home/user/scripts/constants.py")
spark.sparkContext.addPyFile("/home/user/scripts/geodesiccapability.py")
spark.sparkContext.addPyFile("/home/user/scripts/geomath.py")
spark.sparkContext.addPyFile("/home/user/scripts/geodesic.py")
spark.sparkContext.addPyFile("/home/user/scripts/units.py")
spark.sparkContext.addPyFile("/home/user/scripts/util.py")
spark.sparkContext.addPyFile("/home/user/scripts/format.py")
spark.sparkContext.addPyFile("/home/user/scripts/point.py")
spark.sparkContext.addPyFile("/home/user/scripts/distance.py")


first_dataset = spark.read.csv(
    "hdfs://okeanos-master:54310/user/input/Crime_Data_from_2010_to_2019_20231224.csv", header=True, inferSchema=True)
second_dataset = spark.read.csv(
    "hdfs://okeanos-master:54310/user/input/Crime_Data_from_2020_to_Present_20231224.csv", header=True, inferSchema=True)

dataset_df = first_dataset.union(second_dataset)

dataset_df = dataset_df.withColumn("Date Rptd", F.to_timestamp(
    "Date Rptd", "MM/dd/yyyy hh:mm:ss a").cast("date"))
dataset_df = dataset_df.withColumn("DATE OCC", F.to_timestamp(
    "DATE OCC", "MM/dd/yyyy hh:mm:ss a").cast("date"))

policedept_df = spark.read.csv(
    "hdfs://okeanos-master:54310/user/input/LAPD_Police_Stations.csv", header=True, inferSchema=True)

dataset_filt_df = (
    dataset_df
    .filter(F.col("LAT") != 0.0)
    .filter(F.col("Weapon Used Cd").between(100, 199))
    .select(["AREA ", "LAT", "LON"])
)

policedept_filt_df = (
    policedept_df
    .select(["DIVISION", "X", "Y", "PREC"])
    .withColumnRenamed("DIVISION", "division")
)

joined_df = (
    dataset_filt_df
    .join(policedept_filt_df.hint("merge"), dataset_filt_df["AREA "] == policedept_filt_df["PREC"], "inner")
)

print("AREA - POLICE_DEP JOIN: MERGE")
joined_df.explain()


def dist(x1, y1, x2, y2):
    return distance.geodesic((x1, y1), (x2, y2)).km


dist_udf = F.udf(dist, FloatType())

final_df = (
    joined_df
    .withColumn("Distance", dist_udf(F.col("Y"), F.col("X"), F.col("LAT"), F.col("LON")))
    .groupBy("division")
    .agg(F.avg("Distance").alias("average_distance"),
         F.count("division").alias("#")
         )
    .orderBy(F.col("#").desc())
)

final_df.show()

spark.stop()
