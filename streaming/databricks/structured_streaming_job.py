from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, lit


def build_session() -> SparkSession:
    return SparkSession.builder.appName('cold-start-defect-detector').getOrCreate()


def main() -> None:
    spark = build_session()
    stream_df = spark.readStream.format('cloudFiles').option('cloudFiles.format', 'binaryFile').load('dbfs:/mnt/frames')
    anomaly_df = stream_df.select(current_timestamp().alias('timestamp'), lit(0.0).alias('mse_score'), lit('pending').alias('status'))
    query = anomaly_df.writeStream.format('delta').outputMode('append').option('checkpointLocation', 'dbfs:/checkpoints/anomalies').start('dbfs:/delta/anomalies')
    query.awaitTermination()


if __name__ == '__main__':
    main()
