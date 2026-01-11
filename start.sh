#!/bin/sh

set -e

if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

source venv/Scripts/activate

pip install --upgrade pip
pip install -r requirements.txt

docker-compose -f docker/docker-compose.yml up -d

sleep 5

python3 src/producer.py