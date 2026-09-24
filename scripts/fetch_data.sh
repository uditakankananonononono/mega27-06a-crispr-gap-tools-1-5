#!/usr/bin/env bash
# Live data pulls - run outside CI. Sources are public, no auth required.
set -euo pipefail
cd "$(dirname "$0")/../data"
curl -sS -o FC_plus_RES_withPredictions.csv \
  "https://raw.githubusercontent.com/MicrosoftResearch/azimuth/master/azimuth/data/FC_plus_RES_withPredictions.csv"
curl -sS -o V1_suppl_data.txt \
  "https://raw.githubusercontent.com/MicrosoftResearch/azimuth/master/azimuth/data/V1_suppl_data.txt"
curl -sSL -o crisprsql_100720.zip "https://www.crisprsql.com/downloads/100720.zip"
mkdir -p crisprsql && (cd crisprsql && unzip -o ../crisprsql_100720.zip)
echo "Fetched Doench2016 (V1, FC+RES) + crisprSQL 100720."
