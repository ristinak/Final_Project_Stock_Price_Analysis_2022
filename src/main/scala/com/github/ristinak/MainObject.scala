package com.github.ristinak

import com.github.ristinak.SparkUtil.{getSpark, readDataWithView}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{asc, avg, col, count, dense_rank, desc, expr, lit, max, mean, min, rank, round, sqrt, stddev, sum, to_date, to_timestamp}

object MainObject extends App {

  println("Starting the final project")

  val spark = getSpark("Sparky")

  val dfOriginal = readDataWithView(spark, "src/resources/stock_prices_.csv")
  val dailyReturn = round(expr("(close - open)/open * 100"), 2)
  val df = dfOriginal.withColumn("dailyReturn_%", dailyReturn)

  // Daily returns of all stocks by date
  println("Daily returns of all stocks by date:")
  df.orderBy("date").select("date", "ticker", "dailyReturn_%").show(10, false)

  // Average daily return of every stock
  println("Average daily return of every stock:")
  val avgDailyReturn = round(avg(col("dailyReturn_%")),2).as("avgDailyReturn_%")
  df.groupBy("ticker").agg(avgDailyReturn).show(20, false)

  // Average daily return of all stocks by date
  println("Average daily return of all stocks by date:")
  val dfAvgReturn = df.groupBy("date").agg(avgDailyReturn)
  dfAvgReturn.orderBy("date").show(20, false)

  // Most frequently traded stocks on any one day
  println("Most frequently traded stocks on a given day:")
  val frequency = col("volume") * col("close")
  val dfFreq = df.withColumn("frequency", frequency)
  dfFreq.orderBy(desc("frequency")).show(20, false)

  // Most frequently traded stocks on average
  println("Most frequently traded stocks on average:")
  dfFreq.groupBy("ticker")
    .agg(sum("frequency").as("sumFrequency"), avg("frequency").as("avgFrequency"))
    .select("ticker", "sumFrequency", "avgFrequency")
    .orderBy(desc("avgFrequency"))
    .show(10, false)

  // Standard deviation of daily returns
  println("Standard deviation of daily returns:")
  dfAvgReturn.agg(stddev("avgDailyReturn_%")).show()

  // Apple standard deviation
  println("Apple standard deviation:")
  val dfAAPL = df.where(col("ticker")==="AAPL")
  val aaplStdDev = dfAAPL.agg(stddev("dailyReturn_%").as("Standard_deviation_%"))
    .withColumn("Annualized_standard_deviation_%", col("Standard_deviation_%") * lit(15.8745))
  aaplStdDev.show()
//  println("Apple annualized standard deviation:")
//  val aaplStdDevAnnual = expr(aaplStdDev * lit(15.8745079))
//  aaplStdDevAnnual.show()


//  val windowSpec = Window
//    .partitionBy("date")
//    .rowsBetween(Window.unboundedPreceding, Window.currentRow)
//
//  val maxVolume = max(col("volume")).over(windowSpec)
//  val avgReturnAll = avg(col("dailyReturn")).over(windowSpec)

//  println("df with date, max volume, and avgReturn_All:")
//  df.orderBy("date")
////    .groupBy("date")
//    .withColumn("avgReturn_All", avgReturnAll)
//    .withColumn("maxVolume", maxVolume)
//    .select("date", "maxVolume", "avgReturn_All")
//    .show(10, false)
//
//  df.orderBy("date").groupBy("date").sum("dailyReturn").show(20, false)

//  df.withColumn("avgReturn_All", avgReturnAll).show(10, false)

//  val dfRolledUp = df.rollup("date", "ticker")
////    .agg(avg("dailyReturn").as("avgDailyReturn"))
//    .select("date", "ticker", "dailyReturn")
//    .orderBy("date", "ticker")

//  println("dfRolledUp:")
//  dfRolledUp.show(20, false)


}
