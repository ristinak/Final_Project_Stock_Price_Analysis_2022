package com.github.ristinak
import com.github.ristinak.SparkUtil.{getSpark, readDataWithView}
import org.apache.spark.sql.functions.{avg, col, desc, expr, lit, percent_rank, round, sqrt, stddev, sum, to_date, to_timestamp, when}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{OneHotEncoder, RFormula, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.Window
object Changes extends App {

  def LogisticPredictor(df: DataFrame): Unit = {

    val newDF = df.withColumn("change",
      when(col("dailyReturn_%") > 0, "UP")
        when(col("dailyReturn_%") < 0, "DOWN")
        when(col("dailyReturn_%") === 0, "UNCHANGED"))

    val rankDF = newDF.withColumn("rank", percent_rank().over(Window.partitionBy("ticker").orderBy("date")))
    val train = rankDF.where("rank <= 0.7").drop("rank")
    val test = rankDF.where("rank > 0.7").drop("rank")

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
      .setMetricName("accuracy") //different Evaluators will have different metric options
      .setPredictionCol("prediction")
      .setLabelCol("label")

    val tvs = new TrainValidationSplit()
      .setTrainRatio(0.7) // the default is 0.75
      .setEstimatorParamMaps(params) //so this is grid of different hyperparameters
      .setEstimator(pipeline) //these are the various tasks we want done /transformations /
      .setEvaluator(evaluator) //and this is the metric to judge our success

    val tvsFitted = tvs.fit(train) // fitting/making the best model
    val tvsTransformed = tvsFitted.transform(test)

    println(s"Model accuracy according to Multiclass Classification Evaluator: ${evaluator.evaluate(tvsTransformed)}")
    println("Prediction and how it compares to the real data:")
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
  }

}
