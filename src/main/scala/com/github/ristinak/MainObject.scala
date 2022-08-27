package com.github.ristinak

import com.github.ristinak.SparkUtil.{getSpark, readDataWithView}
import org.apache.spark.sql.functions.{avg, col, desc, expr, lit, percent_rank, round, sqrt, stddev, sum, to_date, to_timestamp, when}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.{OneHotEncoder, RFormula, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.Window


// TODO: write scaladoc

object MainObject extends App {

  // *** Getting the dataframe from file and preparing it for analysis ***

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

  // *** Calling our methods ***

  showAverages(df, 10, saveAsCSV = true)

  // *** Bonus Question ***
  // Average and annualized average standard deviation of daily returns (volatility)

  showVolatility(df, saveAsCSV = true)

  // *** Big Bonus Part 1 ***
  // Build a model trying to predict the next day's UP/DOWN/UNCHANGED classificator

  LogisticPredictor(df, 30)

  // *** Big Bonus Part 2 ***
  // Build a model trying to predict the next day's closing price

  LinearRegressionModel(df)

  def showAverages(df: DataFrame, printLines: Int = 20, saveAsParquet: Boolean = true, saveAsCSV: Boolean = false): Unit = {

    // Daily returns of all stocks by date
    println("Daily returns of all stocks by date:")
    df.orderBy("date").select("date", "ticker", "dailyReturn_%").show(10, false)

    // Average daily return of every stock
    println("Average daily return of every stock:")
    val avgDailyReturn = round(avg(col("dailyReturn_%")),2).as("avgDailyReturn_%")
    df.groupBy("ticker").agg(avgDailyReturn).show(printLines, false)

    // Average daily return of all stocks by date
    println("Average daily return of all stocks by date:")
    val dfAvgReturn = df.groupBy("date").agg(avgDailyReturn.as("average_return")).orderBy("date")
    dfAvgReturn.show(printLines, false)

    if (saveAsParquet) write2Parquet(dfAvgReturn, filepath = "src/resources/parquet/averages.parquet")
    if (saveAsCSV) write2CSV(dfAvgReturn, filepath = "src/resources/csv/averages.csv")

    // Most frequently traded stocks on any one day
    println("Most frequently traded stocks on a given day:")
    val frequency = col("volume") * col("close")
    val dfFreq = df.withColumn("frequency", frequency)
    dfFreq.orderBy(desc("frequency")).show(printLines, false)

    // Most frequently traded stocks on average
    println("Most frequently traded stocks on average:")
    dfFreq.groupBy("ticker")
      .agg(sum("frequency").as("sumFrequency"), avg("frequency").as("avgFrequency"))
      .select("ticker", "sumFrequency", "avgFrequency")
      .orderBy(desc("avgFrequency"))
      .show(printLines, false)
  }

  def showVolatility(df: DataFrame, printLines: Int = 20, saveAsParquet: Boolean = true, saveAsCSV: Boolean = false): Unit = {

    println("Stocks ordered by annualized volatility, %:")

    val volatility = round(stddev("dailyReturn_%"),2)
    val annVolatility = round(col("Volatility") * sqrt(lit(252)),2)
    val stdDevDF = df.groupBy("ticker").agg(volatility.as("Volatility"))
      .withColumn("Annualized_Volatility", annVolatility)

    stdDevDF.orderBy(desc("Annualized_Volatility")).show(printLines, false)

    if (saveAsParquet) write2Parquet(stdDevDF, filepath = "src/resources/parquet/volatility.parquet")
    if (saveAsCSV) write2CSV(stdDevDF, filepath = "src/resources/csv/volatility.csv")

  }

  def write2Parquet(df: DataFrame, filepath: String = "src/resources/parquet/savedFile.parquet"): Unit = {
    df.write.mode("overwrite").option("header", true).parquet(filepath)
  }

  def write2CSV(df: DataFrame, filepath: String = "src/resources/parquet/savedFile.csv"): Unit = {
    df.write.mode("overwrite").option("header", true).csv(filepath)
  }

  def LogisticPredictor(df: DataFrame, printLines:Int = 50): Unit = {

    val newDF = df.withColumn("change",
      when(col("dailyReturn_%") > 0, "UP")
        when(col("dailyReturn_%") < 0, "DOWN")
        when(col("dailyReturn_%") === 0, "UNCHANGED"))

    val rankDF = newDF.withColumn("rank", percent_rank().over(Window.partitionBy("ticker").orderBy("date")))
    val train = rankDF.where("rank <= 0.7").drop("rank")
    val test = rankDF.where("rank > 0.7").drop("rank")

    train.orderBy("date").show(30)
    test.orderBy("date").show(30)

    val rForm = new RFormula()
    val logisticReg = new LogisticRegression()
    val stages = Array(rForm, logisticReg)
    val pipeline = new Pipeline().setStages(stages)

    val params = new ParamGridBuilder()
      .addGrid(rForm.formula, Array("change ~ ."))
      .addGrid(logisticReg.elasticNetParam, Array(0.0, 0.5, 1.0))
      .addGrid(logisticReg.regParam, Array(0.1, 2.0))
      .build()

    val evaluator = new MulticlassClassificationEvaluator() // we have 3 possible outcomes
      .setMetricName("accuracy")
      .setPredictionCol("prediction")
      .setLabelCol("label")

    val tvs = new TrainValidationSplit()
      .setTrainRatio(0.7) // the default is 0.75
      .setEstimatorParamMaps(params) //so this is grid of different hyperparameters
      .setEstimator(pipeline) //these are the various tasks we want done /transformations /
      .setEvaluator(evaluator) //and this is the metric to judge our success

    val tvsFitted = tvs.fit(train) // fitting/making the best model
    val tvsTransformed = tvsFitted.transform(test)

    println("Prediction and how it compares to the real data:")
    tvsTransformed
      .select("date", "open", "close", "volume", "ticker", "dailyReturn_%", "change", "label", "prediction", "probability")
      .orderBy("date")
      .show(printLines, false)
    println(s"Logistic Regression model accuracy according to Multiclass Classification Evaluator: ${evaluator.evaluate(tvsTransformed)}\n")

    val trainedPipeline = tvsFitted.bestModel.asInstanceOf[PipelineModel]
    val TrainedLR = trainedPipeline.stages(1).asInstanceOf[LogisticRegressionModel]
    val summaryLR = TrainedLR.summary
    summaryLR.objectiveHistory

    tvsFitted.write.overwrite().save("src/resources/tmp/modelLocation")
  }

  def LinearRegressionModel(df: DataFrame, printLines:Int = 20): Unit = {

    println("Linear Regression Model:")

    val dfRegr = df.withColumn("volume", col("volume").cast("double"))
      .withColumn("date", col("date").cast("string"))

    dfRegr.printSchema()
    dfRegr.show(5, false)
    dfRegr.describe().show(false)

    val indexedDate = new StringIndexer()
      .setInputCol("date")
      .setOutputCol("indexedDate")

    val indexedDateDfRegr = indexedDate.fit(dfRegr).transform(dfRegr)

    val encoder = new OneHotEncoder()
      .setInputCol("indexedDate")
      .setOutputCol("encodedIndexedDate")

    val indexedTicker = new StringIndexer()
      .setInputCol("ticker")
      .setOutputCol("indexedTicker")

    val vecAssembler = new VectorAssembler()
      .setInputCols(Array("encodedIndexedDate", "open", "close", "high", "low", "volume", "indexedTicker", "dailyReturn_%"))
      .setOutputCol("features")

    val linearRegression = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("close")

    val paramGrid = new ParamGridBuilder()
      .addGrid(linearRegression.regParam, Array(0.1, 0.3, 0.5, 0.7))
      .build()

    val stages = Array(encoder, indexedTicker, vecAssembler, linearRegression)
    val pipeline = new Pipeline().setStages(stages)

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse") //rootMeanSquaredError
      .setLabelCol("close")
      .setPredictionCol("prediction")

    val tvs = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.75)
      .setEvaluator(evaluator)

    val rankDF = indexedDateDfRegr.withColumn("rank", percent_rank().over(Window.partitionBy("ticker").orderBy("date")))
    val train = rankDF.where("rank <= 0.7").drop("rank")
    val test = rankDF.where("rank > 0.7").drop("rank")

    val modelLR = tvs.fit(train)
    val predictionLR = modelLR.transform(test)

    predictionLR.select("date", "volume", "ticker", "dailyReturn_%", "open", "close", "prediction").show(printLines, false)
//      predictionLR.show(printLines, false)

    val metrics = predictionLR.select("prediction", "close")
    val rm = new RegressionMetrics(metrics.rdd.map(x =>
      (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

    println("Test data metrics:")
    println("MAE: " + rm.meanAbsoluteError) //MAE measures the average magnitude of the errors in a set of predictions, without considering their direction.
    println("RMSE: " + rm.rootMeanSquaredError) //RMSE represents the square root of the variance of the residuals, the smaller number, the better
    println("R Squared: " + rm.r2) //R-Squared value of 0.9 would indicate that 90% of the variance of the dependent variable being studied is explained by the variance of the independent variable

    val trainedPipeline = modelLR.bestModel.asInstanceOf[PipelineModel]
    val trainedLR = trainedPipeline.stages(3).asInstanceOf[LinearRegressionModel]

    trainedLR.summary.objectiveHistory

    modelLR.write.overwrite().save("src/resources/tmp/linearRegressionModelLocation")
  }


}
