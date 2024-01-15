from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType

spark = SparkSession.builder.appName("part5-df-merge").getOrCreate()

first_dataset = spark.read.csv(
    "hdfs://okeanos-master:54310/user/input/Crime_Data_from_2010_to_2019_20231224.csv", header=True, inferSchema=True)
second_dataset = spark.read.csv(
    "hdfs://okeanos-master:54310/user/input/Crime_Data_from_2020_to_Present_20231224.csv", header=True, inferSchema=True)

dataset_df = first_dataset.union(second_dataset)

dataset_df = dataset_df.withColumn("Date Rptd", F.to_timestamp(
    "Date Rptd", "MM/dd/yyyy hh:mm:ss a").cast("date"))
dataset_df = dataset_df.withColumn("DATE OCC", F.to_timestamp(
    "DATE OCC", "MM/dd/yyyy hh:mm:ss a").cast("date"))

income2015_df = spark.read.csv(
    "hdfs://okeanos-master:54310/user/input/income/LA_income_2015.csv", header=True, inferSchema=True)
revgeocoding_df = spark.read.csv(
    "hdfs://okeanos-master:54310/user/input/revgecoding.csv", header=True, inferSchema=True)

dataset_filt_df = (
    dataset_df
    .select(["DATE OCC", "Vict Descent", "LAT", "LON"])
)


def descent_description(descent):
    if descent == "A":
        return "Other Asian"
    if descent == "B":
        return "Black"
    if descent == "C":
        return "Chinese"
    if descent == "D":
        return "Cambodian"
    if descent == "F":
        return "Filipino"
    if descent == "G":
        return "Guamanian"
    if descent == "H":
        return "Hispanic/Latin/Mexican"
    if descent == "I":
        return "American Indian/Alaskan Native"
    if descent == "J":
        return "Japanese"
    if descent == "K":
        return "Korean"
    if descent == "L":
        return "Laotian"
    if descent == "O":
        return "Other"
    if descent == "P":
        return "Pacific Islander"
    if descent == "S":
        return "Samoan"
    if descent == "U":
        return "Hawaiian"
    if descent == "V":
        return "Vietnamese"
    if descent == "W":
        return "White"
    if descent == "Z":
        return "Asian Indian"


descript_udf = F.udf(descent_description, StringType())


def get_income(string_income):
    return int(string_income[1:].replace(",", ""))


get_income_udf = F.udf(get_income, IntegerType())

df_descent_zip = (
    dataset_df
    .withColumn("year", F.year("DATE OCC"))
    .filter("year == 2015")
    .filter((F.col("Vict Descent") != "X") & (F.col("Vict Descent").isNotNull()))
    .join(revgeocoding_df.hint("merge"), (revgeocoding_df["LAT"] == dataset_df["LAT"]) & (revgeocoding_df["LON"] == dataset_df["LON"]), "inner")
    .withColumn("Zip code", F.col("ZIPcode").substr(1, 5).cast("int"))
    .select(["Vict Descent", "Zip code"])
)

print("CRIME - ZIP JOIN: MERGE")
df_descent_zip.explain()

df_descent_income = (
    df_descent_zip
    .join(income2015_df.hint("merge"), income2015_df["Zip Code"] == df_descent_zip["Zip code"])
    .select(["Vict Descent", "Estimated Median Income"])
    .withColumn("Victim Descent", descript_udf(F.col("Vict Descent")))
)

print("CRIME - INCOME JOIN: MERGE")
df_descent_income.explain()

df_descent_income = (
    df_descent_income
    .select(["Victim Descent", "Estimated Median Income"])
    .withColumn("Estimated Median Income", get_income_udf(F.col("Estimated Median Income")))
)

income_values = (
    df_descent_income
    .groupBy("Estimated Median Income")
    .agg(F.count("*").alias("#"))
    .orderBy(F.col("Estimated Median Income").desc())
    .collect()
)

all_income_descent = (
    df_descent_income
    .filter((F.col("Estimated Median Income") == income_values[0][0])
            | (F.col("Estimated Median Income") == income_values[1][0])
            | (F.col("Estimated Median Income") == income_values[2][0])
            | (F.col("Estimated Median Income") == income_values[-1][0])
            | (F.col("Estimated Median Income") == income_values[-2][0])
            | (F.col("Estimated Median Income") == income_values[-3][0]))
    .groupBy("Victim Descent")
    .agg(F.count("*").alias("#"))
    .orderBy(F.col("#").desc())
)

all_income_descent.show()

spark.stop()
