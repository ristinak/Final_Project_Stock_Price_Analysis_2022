package com.github.ristinak

import com.github.ristinak.SparkUtil.{getSpark, readDataWithView}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{asc, avg, col, desc, expr, lag, lit, round, sqrt, stddev, sum, to_date, to_timestamp, when}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, RFormula, StringIndexer}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{ColumnName, TypedColumn}

object MainObject extends App {

  println("Starting the final project")
  val spark = getSpark("Sparky")

  // saving the dataframe and dropping any null values
  val dfOriginal = readDataWithView(spark, "src/resources/stock_prices_.csv").na.drop("any")

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
  val avgDailyReturn = round(avg(col("dailyReturn_%")),3).as("avgDailyReturn_%")
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

  val newDF = df.withColumn("change",
      when(col("dailyReturn_%") > 0, "UP")
    when(col("dailyReturn_%") < 0, "DOWN")
  when(col("dailyReturn_%") === 0, "UNCHANGED")
  )

  println("************* The newDF with the column 'change': **************")
  newDF.show(10, false)


  // trying the new indexing and encoding of strings //
  val indexer = new StringIndexer()
    .setInputCols(Array("ticker", "change"))
    .setOutputCols(Array("ticker_index", "change_index"))

  val encoder = new OneHotEncoder()
    .setInputCols(Array("ticker_index", "change_index"))
    .setOutputCols(Array("ticker_encoded", "change_encoded"))

//  // Indexing the ticker column in order to include it in the feature
//  val tickerInd = new StringIndexer().setInputCol("ticker").setOutputCol("tickerInd")
//  val tickerIndDF = tickerInd.fit(newDF).transform(newDF)
//  println("************* The newDF with ticker index: **************")
//  tickerIndDF.show(10, false)
//
//  // Indexing the change column
//  val chgInd = new StringIndexer().setInputCol("change").setOutputCol("changeInd")
//  val changeIndDF = chgInd.fit(tickerIndDF).transform(tickerIndDF)
//  println("************* The newDF with change index: **************")
//  changeIndDF.show(10, false)

//  val Array(train, test) = changeIndDF.randomSplit(Array(0.7, 0.3))
  val Array(train, test) = newDF.randomSplit(Array(0.7, 0.3))

  val rForm = new RFormula()
  val lr = new LogisticRegression()
//  val stages = Array(rForm, lr)
  val stages = Array(indexer, encoder, rForm, lr)
  val pipeline = new Pipeline().setStages(stages)


  // (Test Evaluation,0.997)
  val params = new ParamGridBuilder()
    .addGrid(rForm.formula, Array(
      "change_index ~ ."))
    .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
    .addGrid(lr.regParam, Array(0.1, 2.0))
    .build()

  val evaluator = new BinaryClassificationEvaluator()
    .setMetricName("areaUnderROC") //different Evaluators will have different metric options
    .setRawPredictionCol("prediction")
    .setLabelCol("label")

  val tvs = new TrainValidationSplit()
    .setTrainRatio(0.7) // the default is 0.75
    .setEstimatorParamMaps(params) //so this is grid of different hyperparameters
    .setEstimator(pipeline) //these are the various tasks we want done /transformations /
    .setEvaluator(evaluator) //and this is the metric to judge our success

  val tvsFitted = tvs.fit(train) // fitting/making the best model

  //And of course evaluate how it performs on the test set!
  println("Test Evaluation", evaluator.evaluate(tvsFitted.transform(test)))
  println("Let's look at the prediction and how it compares to the real data:")
  tvsFitted.transform(test).show(20, false)

  val trainedPipeline = tvsFitted.bestModel.asInstanceOf[PipelineModel]
  val TrainedLR = trainedPipeline.stages(3).asInstanceOf[LogisticRegressionModel]
  val summaryLR = TrainedLR.summary
  summaryLR.objectiveHistory
  //Persisting and Applying Models
  //Now that we trained this model, we can persist it to disk to use it for prediction purposes later on:
  tvsFitted.write.overwrite().save("src/resources/tmp/modelLocation")


}
