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
    .collect()
)

policedept_filt_df = (
    policedept_df
    .select(["DIVISION", "X", "Y"])
    .withColumnRenamed("DIVISION", "division")
)

def dist(x1, y1, x2, y2):
    return distance.geodesic((x1, y1), (x2, y2)).km

dist_udf = F.udf(dist, FloatType())

schema = StructType([
    StructField("division", StringType(), True),
    StructField("Distance", FloatType(), True)
])

mindist_df = spark.createDataFrame([], schema=schema)

loopCount = len(dataset_filt_df)
for i in range(loopCount):
    crime_row = dataset_filt_df[i]
    crime_df = spark.createDataFrame([crime_row])
    joined_df = policedept_filt_df.crossJoin(crime_df)
    dist_df = joined_df.withColumn("Distance", dist_udf(F.col("Y"), F.col("X"), F.col("LAT"), F.col("LON"))).collect()

    minDist = 100   # large enough value
    closestDept = ""
    for j in range(joined_df.count()):
        currDist = dist_df[j]["Distance"]
        if currDist < minDist:
            minDist = currDist
            closestDept = dist_df[j]["division"]

    new_row = Row(division=closestDept, Distance=minDist)
    mindist_df = mindist_df.union(spark.createDataFrame([new_row]))

final_df = (
    mindist_df
    .groupBy("division")
    .agg(F.avg("Distance").alias("average_distance"), 
         F.count("division").alias("#")
    )
    .orderBy(F.col("#").desc())
)

final_df.show()

spark.stop()