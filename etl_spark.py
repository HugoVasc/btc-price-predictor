from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, expr, year, month
from pyspark.sql.window import Window
import yfinance as yf

# Inicializar a SparkSession
spark = SparkSession.builder.appName("BTC-USD Analysis").getOrCreate()

# GET BTC - USD OHLCV DATA
btc = yf.Ticker('BTC-USD')
btc_hist = btc.history(start='2022-01-01').reset_index()

# Converter para Spark DataFrame
btc_spark_df = spark.createDataFrame(btc_hist)

# Criar colunas de ano e mês para particionamento
btc_spark_df = btc_spark_df \
    .withColumn("year", year(col("Date"))) \
    .withColumn("month", month(col("Date")))

# Calcular as médias móveis de 7, 25 e 99 dias
window_7 = Window.orderBy("Date").rowsBetween(-6, 0)
window_25 = Window.orderBy("Date").rowsBetween(-24, 0)
window_99 = Window.orderBy("Date").rowsBetween(-98, 0)

btc_spark_df = btc_spark_df \
    .withColumn("m_avg_7", avg(col("Close")).over(window_7)) \
    .withColumn("m_avg_25", avg(col("Close")).over(window_25)) \
    .withColumn("m_avg_99", avg(col("Close")).over(window_99))

# Adicionar colunas de variação percentual
btc_spark_df = btc_spark_df \
    .withColumn("close_diff", expr("Close / lag(Close, 1) over (order by Date) - 1")) \
    .withColumn("m_avg_7_diff", expr("m_avg_7 / lag(m_avg_7, 1) over (order by Date) - 1")) \
    .withColumn("m_avg_25_diff", expr("m_avg_25 / lag(m_avg_25, 1) over (order by Date) - 1")) \
    .withColumn("m_avg_99_diff", expr("m_avg_99 / lag(m_avg_99, 1) over (order by Date) - 1"))

# Remover colunas indesejadas e valores nulos
btc_spark_df = btc_spark_df.drop("Dividends", "Stock Splits").na.drop()

# Salvar como parquet com particionamento por ano e mês
btc_spark_df.write.mode("overwrite").parquet("./btc_hist_partitioned.parquet")
