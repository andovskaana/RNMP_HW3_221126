import csv
import os
from typing import List, Tuple

from sklearn.model_selection import train_test_split

def read_data(filepath: str) -> Tuple[List[str], List[List[str]]]:
    """Читање на CSV датотеката и враќање на header и редовите"""
    with open(filepath, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]
    return header, rows


def write_data(filepath: str, header: List[str], rows: List[List[str]]) -> None:

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def stratified_split(
    rows: List[List[str]], test_size: float = 0.2, random_state: int = 42
) -> Tuple[List[List[str]], List[List[str]]]:
    """Поделба на податочното множество на:
            offline.csv (80% од големината на оригиналното множество)
            online.csv (20% од големината на оригиналното множество)
       Се задржуваат соодносот на класите (стратификација).
    """
    labels = [row[0] for row in rows]
    X_train, X_test, _, _ = train_test_split(
        rows, labels, test_size=test_size, stratify=labels, random_state=random_state
    )
    return X_train, X_test


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    original_path = os.path.join(data_dir, "diabetes_binary_health_indicators_BRFSS2015.csv")
    offline_path = os.path.join(data_dir, "offline.csv")
    online_path = os.path.join(data_dir, "online.csv")

    header, rows = read_data(original_path)

    offline_rows, online_rows = stratified_split(rows)

    write_data(offline_path, header, offline_rows)
    write_data(online_path, header, online_rows)
    print(f"Wrote {len(offline_rows)} rows to {offline_path}")
    print(f"Wrote {len(online_rows)} rows to {online_path}")


if __name__ == "__main__":
    main()