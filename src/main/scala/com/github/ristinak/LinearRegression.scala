package com.github.ristinak

import com.github.ristinak.SparkUtil.{getSpark, readDataWithView}
import org.apache.spark.sql.functions.{avg, col, desc, expr, lit, round, sqrt, stddev, sum, to_date, to_timestamp, when}
import org.apache.spark.ml.classification.{DecisionTreeClassifier, LogisticRegression, LogisticRegressionModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.{OneHotEncoder, RFormula, StandardScaler, StringIndexer, Tokenizer, VectorAssembler}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.{Pipeline, PipelineModel, classification}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel, RandomForestRegressionModel}
import org.apache.spark.sql.DataFrame

object LinearRegression {

  def main(args: Array[String]): Unit = {

    val filePath = if (!args.isEmpty) args(0) else "src/resources/stock_prices_.csv"

    println("Starting the final project")
    val spark = getSpark("Sparky")

    // saving the dataframe and dropping any null values
    val dfOriginal = readDataWithView(spark, filePath).na.drop("any")

    // converting date to fit the format yyyy-MM-dd
    spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
    val dfWithDate = dfOriginal
      .withColumn("date", to_date(col("date"), "yyyy-MM-dd"))

    // adding column dailyReturn_% to our dataframe
    val dailyReturn = round(expr("(close - open)/open * 100"), 4)
    val df = dfWithDate.withColumn("dailyReturn_%", dailyReturn)

    // saving this extended dataframe to a parquet file
    df.write.mode("overwrite").parquet("src/resources/parquet/daily-returns.parquet")

    // Daily returns of all stocks by date
    println("Daily returns of all stocks by date:")
    df.orderBy("date").select("date", "ticker", "dailyReturn_%").show(10, false)

    // Average daily return of every stock
    println("Average daily return of every stock:")
    val avgDailyReturn = round(avg(col("dailyReturn_%")),2).as("avgDailyReturn_%")
    df.groupBy("ticker").agg(avgDailyReturn).show(20, false)

    // Average daily return of all stocks by date
    println("Average daily return of all stocks by date:")
    val dfAvgReturn = df.groupBy("date").agg(avgDailyReturn.as("average_return")).orderBy("date")
    dfAvgReturn.show(20, false)
    // saving daily return averages to a parquet file
    dfAvgReturn.write.mode("overwrite").parquet("src/resources/parquet/average_return.parquet")

    // Most frequently traded stocks on any one day
    println("Most frequently traded stocks on a given day:")
    val frequency = col("volume") * col("close")
    val dfFreq = df.withColumn("frequency", frequency)
    dfFreq.orderBy(desc("frequency")).show(10, false)

    // Most frequently traded stocks on average
    println("Most frequently traded stocks on average:")
    dfFreq.groupBy("ticker")
      .agg(sum("frequency").as("sumFrequency"), avg("frequency").as("avgFrequency"))
      .select("ticker", "sumFrequency", "avgFrequency")
      .orderBy(desc("avgFrequency"))
      .show(10, false)

    println("Read the parquet file of average daily returns:")
    spark.read.parquet("src/resources/parquet/average_return.parquet").show(20, false)


    // *** Bonus Question ***
    // Average and annualized average standard deviation of daily returns (volatility)
    println("*** Bonus Question ***")
    println("Stocks ordered by annualized volatility, %:")

    val volatility = round(stddev("dailyReturn_%"),2)
    val annVolatility = round(col("Volatility") * sqrt(lit(252)),2)
    val stdDevDF = df.groupBy("ticker").agg(volatility.as("Volatility"))
      .withColumn("Annualized_Volatility", annVolatility)

    stdDevDF.orderBy(desc("Annualized_Volatility")).show()

    // ******************* Building a model *******************

    //Linear regression//

    val dfRegr = df.withColumn("volume", col("volume").cast("double"))
      .withColumn("date", col("date").cast("string"))

    dfRegr.printSchema()
    dfRegr.show(10, false)
    dfRegr.describe().show(false)

    val indexedDate = new StringIndexer()
      .setInputCol("date")
      .setOutputCol("indexedDate")

    val encoder = new OneHotEncoder()
      .setInputCol("indexedDate")
      .setOutputCol("encodedIndexedDate")

    val indexedDateDfRegr = indexedDate.fit(dfRegr).transform(dfRegr)
    val encodedIndexedDateDfRegr = encoder.fit(indexedDateDfRegr).transform(indexedDateDfRegr)

    val indexedTicker = new StringIndexer()
      .setInputCol("ticker")
      .setOutputCol("indexedTicker")

    val indexedDfRegr = indexedTicker.fit(encodedIndexedDateDfRegr).transform(encodedIndexedDateDfRegr)

    val vecAssembler = new VectorAssembler()
      .setInputCols(Array("encodedIndexedDate", "open", "close", "high", "low", "volume", "indexedTicker", "dailyReturn_%"))
      .setOutputCol("features")

    val dfIncVector = vecAssembler.transform(indexedDfRegr)

    //val Array(train, test) = dfIncVector.randomSplit(Array(0.7, 0.3))
    val train = dfIncVector.where(col("date") < "2016-07-20")
    val test = dfIncVector.where(col("date") >= "2016-07-20")

    dfIncVector.show(10, false)

    val lr = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setFeaturesCol("features")
      .setLabelCol("close")

    val lrModel = lr.fit(train)
    val lrPredictions = lrModel.transform(test)

    lrPredictions.show(50, false)

    train.describe().show()
    test.describe().show()

    def showAccuracy(df: DataFrame): Unit = {
      val evaluator = new RegressionEvaluator()
        .setMetricName("rmse") //rootMeanSquaredError
        .setLabelCol("close")
        .setPredictionCol("prediction")
      val accuracy = evaluator.evaluate(df)
      println(s"Accuracy: $accuracy") //hmmm...
    }

    showAccuracy(lrPredictions) // linear regression model

  }}
