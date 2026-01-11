import argparse
import json
from kafka import KafkaConsumer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default="localhost:9092")
    ap.add_argument("--topic", default="health_data_predicted")
    ap.add_argument("--from-beginning", action="store_true")
    args = ap.parse_args()

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=[args.bootstrap],
        auto_offset_reset="earliest" if args.from_beginning else "latest",
        enable_auto_commit=True,
        group_id="diabetes-ml-consumer",
        value_deserializer=lambda b: b.decode("utf-8", errors="replace"),
    )

    print(f"Consuming topic='{args.topic}' @ {args.bootstrap} (from_beginning={args.from_beginning})")
    print("Press Ctrl+C to stop.\n")

    try:
        for msg in consumer:
            raw = msg.value
            try:
                payload = json.loads(raw)
            except Exception:
                print(raw)
                continue

            pred = payload.get("predicted_diabetes", payload.get("prediction"))
            prob = payload.get("prob_diabetes")

            if pred is None:
                print(payload)
                continue

            label = "DIABETES" if int(pred) == 1 else "NO_DIABETES"
            if prob is None:
                print(f"prediction={int(pred)} ({label})")
            else:
                print(f"prediction={int(pred)} ({label}) | prob={float(prob):.4f}")
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
