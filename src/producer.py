import os
import time
import json
import csv
import argparse
from typing import Dict, Generator, Any, Optional

from kafka import KafkaProducer


def load_records(csv_path: str) -> Generator[Dict[str, Any], None, None]:
    """Yield each CSV row as a dict, casting values to float when possible."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record: Dict[str, Any] = {}
            for k, v in row.items():
                if v is None:
                    record[k] = None
                    continue
                v = v.strip()
                # Проба за float cast (за секој случај да се сите податоци бројки)
                try:
                    record[k] = float(v)
                except ValueError:
                    record[k] = v
            yield record


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stream offline.csv records to Kafka as JSON")
    p.add_argument("--bootstrap", default=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
                   help="Kafka bootstrap servers (default: env KAFKA_BOOTSTRAP_SERVERS or localhost:9092)")
    p.add_argument("--topic", default=os.environ.get("TOPIC", "health_data"),
                   help="Kafka topic to produce to (default: env TOPIC or health_data)")
    p.add_argument("--sleep-ms", type=int, default=int(os.environ.get("SLEEP_MS", "100")),
                   help="Sleep between messages in ms (default: env SLEEP_MS or 100)")
    p.add_argument("--max-records", type=int, default=0,
                   help="Send only first N records (0 = all)")
    p.add_argument("--quiet", action="store_true", help="Do not print each produced record")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    offline_path = os.path.join(project_root, "data", "offline.csv")
    if not os.path.exists(offline_path):
        raise FileNotFoundError(f"offline.csv not found at: {offline_path}")

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=0,
        acks="all",
    )

    sent = 0
    for i, record in enumerate(load_records(offline_path), start=1):
        if args.max_records and sent >= args.max_records:
            break

        producer.send(args.topic, value=record)
        # flush ми е ставено на секои 20 записи (Може да се прилагоди како било на колку било)
        if sent % 20 == 0:
            producer.flush()
        sent += 1

        if not args.quiet:
            print(f"[PRODUCED {sent}] topic={args.topic} value={json.dumps(record, ensure_ascii=False)}")

        time.sleep(max(args.sleep_ms, 0) / 1000.0)

    print(f"Finished. Sent {sent} record(s) to {args.topic} on {args.bootstrap}")


if __name__ == "__main__":
    main()
