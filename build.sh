#!/bin/bash
set -eu

docker compose down
docker-compose down
docker-compose build --no-cache
