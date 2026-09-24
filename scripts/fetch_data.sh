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
curl -sSL -o /tmp/aax9249_suppl.zip \
  "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6834390/supplementaryFiles"
unzip -o -j /tmp/aax9249_suppl.zip aax9249_Table_S1.xlsx -d .
echo "Fetched DeepSpCas9 Table S1 (Kim 2019, Sci Adv eaax9249; SRA SRP150719)."
