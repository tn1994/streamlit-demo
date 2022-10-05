#!/bin/bash
set -eu

docker compose down
docker-compose down
docker-compose up
