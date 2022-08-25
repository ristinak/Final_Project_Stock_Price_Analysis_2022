package com.github.ristinak

import com.github.ristinak.SparkUtil.{getSpark, readDataWithView}
import org.apache.spark.sql.functions.{avg, col, desc, expr, lit, round, sqrt, stddev, sum, to_date, to_timestamp, when}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{OneHotEncoder, RFormula, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame

// TODO?: copy-paste the showAccuracy method that Valdis wrote - maybe unnecessary?
// TODO: write a model to predict the closing price (regression)
// TODO: write scaladoc

// DONE: check to see if another csv file can be used through program arguments:
// In order to get the filepath for a different csv file, these steps are needed:
// 1. download any two csv files from https://finance.yahoo.com/trending-tickers
// 2. enter the filepaths for these csvs into csvCombiner.scala program
// 3. change the outputPath value in csvCombiner.scala
// 4. run the program
// 5. the new filepath (outputPath) is now ready to be entered as 'program arguments':
// in IntelliJ click Run -> Edit Configurations... -> enter the filepath into 'Program arguments'


object MainObject {

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

    val newDF = df.withColumn("change",
        when(col("dailyReturn_%") > 0, "UP")
      when(col("dailyReturn_%") < 0, "DOWN")
    when(col("dailyReturn_%") === 0, "UNCHANGED")
    )

    println("************* The newDF with the column 'change': **************")
    newDF.show(10, false)

    val train = newDF.where(col("date") < "2016-07-19")
    val test = newDF.where(col("date") >= "2016-07-19")


    val rForm = new RFormula()
    val logisticReg = new LogisticRegression()
    val stages = Array(rForm, logisticReg)
    val pipeline = new Pipeline().setStages(stages)

    val params = new ParamGridBuilder()
      .addGrid(rForm.formula, Array(
        "change ~ ."))
      .addGrid(logisticReg.elasticNetParam, Array(0.0, 0.5, 1.0))
      .addGrid(logisticReg.regParam, Array(0.1, 2.0))
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
    val tvsTransformed = tvsFitted.transform(test)

    //And of course evaluate how it performs on the test set!
    println("Test Evaluation", evaluator.evaluate(tvsTransformed))
    println("Let's look at the prediction and how it compares to the real data:")
    tvsTransformed
      .select("date", "open", "close", "volume", "ticker", "dailyReturn_%", "change", "label", "prediction", "probability")
      .orderBy("date")
      .show(50, false)

    val trainedPipeline = tvsFitted.bestModel.asInstanceOf[PipelineModel]
    val TrainedLR = trainedPipeline.stages(1).asInstanceOf[LogisticRegressionModel]
    val summaryLR = TrainedLR.summary
    summaryLR.objectiveHistory
    //Persisting and Applying Models
    //Now that we trained this model, we can persist it to disk to use it for prediction purposes later on:
    tvsFitted.write.overwrite().save("src/resources/tmp/modelLocation")


    def showAccuracy(df: DataFrame): Unit = {
      // Select (prediction, true label) and compute test error.
      val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("accuracy")
      val accuracy = evaluator.evaluate(df) //in order for this to work we need label and prediction columns
      println(s"DF size: ${df.count()} Accuracy $accuracy - Test Error = ${(1.0 - accuracy)}")
    }

    showAccuracy(tvsTransformed) // logistic regression


  }}
