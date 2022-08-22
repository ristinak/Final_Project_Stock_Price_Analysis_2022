package com.github.ristinak

import com.github.ristinak.SparkUtil.{getSpark, readDataWithView}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{asc, avg, col, desc, expr, lag, lit, round, sqrt, stddev, sum, to_date, to_timestamp, when}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{ColumnName, TypedColumn}

object MainObject extends App {

  println("Starting the final project")

  val spark = getSpark("Sparky")
  val dfOriginal = readDataWithView(spark, "src/resources/stock_prices_.csv")

  // converting date to fit the format yyyy-MM-dd
  spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
  val dfWithDate = dfOriginal
    .withColumn("date", to_date(col("date"), "yyyy-MM-dd"))

  // adding column dailyReturn_% to our dataframe
  val dailyReturn = round(expr("(close - open)/open * 100"), 2)
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

//    val windowSpec = Window
//      .partitionBy("ticker")
//      .orderBy("date")
//      .rowsBetween(Window.unboundedPreceding, Window.currentRow)
//
//  val priceChange = col("close") - lag("close", 1).over(windowSpec)
//  val newDF = dfWithDate.withColumn("priceChange", priceChange)


  val newDF = df.withColumn("change",
      when(col("dailyReturn_%") > 0, "UP")
    when(col("dailyReturn_%") < 0, "DOWN")
//  when(col("dailyReturn_%") = 0, "UNCHANGED")
  )

  println("************* The newDF with column change: **************")
  newDF.show(10, false)

  val Array(train, test) = newDF.randomSplit(Array(0.7, 0.3))
//  train.describe().show()
  //holdout set we will use test to see how well we did - it is crucial that none of these test data points were used in training
//  test.describe().show()

  val rForm = new RFormula()
  val lr = new LogisticRegression()
  val stages = Array(rForm, lr)
  val pipeline = new Pipeline().setStages(stages)


  // (Test Evaluation,0.7262689691261119)
//  val params = new ParamGridBuilder()
//    .addGrid(rForm.formula, Array(
////      "ticker ~ . + open:close",
//      "ticker ~ open + close",
//      "ticker ~ . + open:high + close:close"))
//    .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
//    .addGrid(lr.regParam, Array(0.1, 2.0))
//    .build()

  // (Test Evaluation,1.0)
  val params = new ParamGridBuilder()
    .addGrid(rForm.formula, Array(
      "ticker ~ close",
      "ticker ~ open + close"))
    .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
    .addGrid(lr.regParam, Array(0.3, 2.0))
    .build()

  val evaluator = new BinaryClassificationEvaluator()
    .setMetricName("areaUnderROC") //different Evaluators will have different metric options
    .setRawPredictionCol("prediction")
    .setLabelCol("label")

  val tvs = new TrainValidationSplit()
    .setTrainRatio(0.75) // also the default.
    .setEstimatorParamMaps(params) //so this is grid of what different hyperparameters
    .setEstimator(pipeline) //these are the various tasks we want done /transformations /
    .setEvaluator(evaluator) //and this is the metric to judge our success

  val tvsFitted = tvs.fit(train) //so this will actually do the work of fitting/making the best model

  //And of course evaluate how it performs on the test set!
  println("Test Evaluation", evaluator.evaluate(tvsFitted.transform(test)))
  println("tvsFitted.transform(test)")
  tvsFitted.transform(test).show(20, false)

  val trainedPipeline = tvsFitted.bestModel.asInstanceOf[PipelineModel]
  val TrainedLR = trainedPipeline.stages(1).asInstanceOf[LogisticRegressionModel]
  val summaryLR = TrainedLR.summary
  summaryLR.objectiveHistory
  //Persisting and Applying Models
  //Now that we trained this model, we can persist it to disk to use it for prediction purposes later on:
  tvsFitted.write.overwrite().save("src/resources/tmp/modelLocation")


}
