#!/bin/sh
echo "Dataset: $1"

if [ $1 == "synthetic_graph_1" ]; then
    mkdir -p "tmp"
    curl --url 'https://syncandshare.lrz.de/dl/fiXfSD14pKGM54L5BqZxF8vF/synthetic_graph_1.zip'\
         --output 'tmp/ds.zip'
elif [ $1 == "CD1_E_no2" ]; then
    mkdir -p "tmp"
    curl --url 'https://syncandshare.lrz.de/dl/fiJf6ukkGCdUQwXBKd4Leusp/CD1-E_no2.zip'\
         --output 'tmp/ds.zip'
else
  echo "Unknown Dataset"
  exit 1
fi

unzip -d "tmp/ds" "tmp/ds.zip"

mkdir -p "datasets/$1"
find tmp/ds -name "*nodes*.csv" -exec mv {} "datasets/$1/nodes.csv" \;
find tmp/ds -name "*edges*.csv" -exec mv {} "datasets/$1/edges.csv" \;

rm -rf tmp