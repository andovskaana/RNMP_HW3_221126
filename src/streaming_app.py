import os
import json
import pickle
from typing import List, Tuple

import numpy as np
from sklearn.utils.parallel import Parallel, delayed
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType


def load_feature_names(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_model_scaler_threshold(
    model_path: str,
    scaler_path: str,
    info_path: str | None = None,
) -> Tuple[object, object, float]:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    threshold = 0.50
    if info_path and os.path.exists(info_path):
        try:
            with open(info_path, "r", encoding="utf-8") as jf:
                info = json.load(jf)
            if isinstance(info, dict) and "threshold" in info:
                threshold = float(info["threshold"])
        except Exception:
            pass

    return model, scaler, threshold


def create_schema(feature_names: List[str]) -> StructType:
    #Дозволување на пораки со Diabetes_binary, подоцна ова не ни треба
    fields = [StructField(name, DoubleType(), True) for name in feature_names]
    if "Diabetes_binary" not in feature_names:
        fields.append(StructField("Diabetes_binary", DoubleType(), True))
    return StructType(fields)


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")

    model_path = os.path.join(models_dir, "best_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    feature_path = os.path.join(models_dir, "feature_names.txt")
    info_path = os.path.join(models_dir, "best_model_info.json")

    feature_names = load_feature_names(feature_path)
    model, scaler, threshold = load_model_scaler_threshold(model_path, scaler_path, info_path=info_path)

    # Precompute scaler params (works for StandardScaler fitted offline)
    mean_arr = np.array(getattr(scaler, "mean_", np.zeros(len(feature_names))), dtype=np.float64)
    scale_arr = np.array(getattr(scaler, "scale_", np.ones(len(feature_names))), dtype=np.float64)
    scale_arr = np.where(scale_arr == 0, 1.0, scale_arr)

    bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    input_topic = os.environ.get("INPUT_TOPIC", "health_data")
    output_topic = os.environ.get("OUTPUT_TOPIC", "health_data_predicted")
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", os.path.join(project_root, "checkpoint"))
    print_console = os.environ.get("PRINT_CONSOLE", "1") not in ("0", "false", "False", "no", "NO")

    spark = SparkSession.builder.appName("DiabetesStreaming").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    schema = create_schema(feature_names)

    raw_df = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe", input_topic)
        .option("startingOffsets", "latest")
        .load()
    )

    parsed_df = (
        raw_df.selectExpr("CAST(value AS STRING) AS json")
        .select(F.from_json(F.col("json"), schema).alias("data"))
        .select("data.*")
    )

    #Ako ne postoi features mu pisuvam 0.0
    features_df = parsed_df.select(
        *[F.coalesce(F.col(c), F.lit(0.0)).cast("double").alias(c) for c in feature_names]
    )

    pred_schema = StructType([
        StructField("prob_diabetes", DoubleType(), nullable=True),
        StructField("predicted_diabetes", DoubleType(), nullable=False),
    ])

    def _predict_row(*cols):
        x = np.array([float(v) for v in cols], dtype=np.float64).reshape(1, -1)
        x_scaled = (x - mean_arr) / scale_arr

        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(x_scaled)[0][1])
            pred = 1.0 if prob >= threshold else 0.0
            return (prob, pred)

        pred = float(model.predict(x_scaled)[0])
        return (None, pred)

    predict_udf = F.udf(_predict_row, returnType=pred_schema)

    enriched_df = (
        features_df
        .withColumn("pred_struct", predict_udf(*[F.col(c) for c in feature_names]))
        .withColumn("prob_diabetes", F.col("pred_struct.prob_diabetes"))
        .withColumn("predicted_diabetes", F.col("pred_struct.predicted_diabetes"))
        .drop("pred_struct")
    )

    # Kafka output: оригинални features + предвидени полиња
    out_cols = feature_names + ["prob_diabetes", "predicted_diabetes"]
    kafka_out = enriched_df.select(F.to_json(F.struct(*[F.col(c) for c in out_cols])).alias("value"))

    kafka_query = (
        kafka_out.writeStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("topic", output_topic)
        .option("checkpointLocation", checkpoint_dir)
        .outputMode("append")
        .start()
    )

    if print_console:
        console_df = enriched_df.select(
            F.col("predicted_diabetes").cast("int").alias("pred"),
            F.when(F.col("predicted_diabetes") == 1, F.lit("DIABETES")).otherwise(F.lit("NO_DIABETES")).alias("label"),
            F.round(F.col("prob_diabetes"), 4).alias("prob"),
        )
        console_df.writeStream.format("console").outputMode("append") \
            .option("truncate", "false").option("numRows", 20) \
            .option("checkpointLocation", checkpoint_dir + "_console").start()

    print(f"[Spark] Reading '{input_topic}' -> writing '{output_topic}' @ {bootstrap_servers}")
    print(f"[Spark] threshold={threshold:.3f} | PRINT_CONSOLE={int(print_console)}")
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
