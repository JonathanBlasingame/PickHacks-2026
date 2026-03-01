#!/usr/bin/env bash
set -euo pipefail

command -v wget >/dev/null || { echo "Please install wget"; exit 1; }
command -v unzip >/dev/null || { echo "Please install unzip"; exit 1; }
command -v tar >/dev/null || { echo "Please install tar"; exit 1; }

ROOT_DIR="${1:-data/datasets}"
mkdir -p "$ROOT_DIR"
cd "$ROOT_DIR"

echo "Downloading CNRPark/PKLot datasets into: $PWD"

# 1) CNRPark patches
if [ ! -d "CNRPark-Patches-150x150" ]; then
  wget -q "http://cnrpark.it/dataset/CNRPark-Patches-150x150.zip" -O "CNRPark.zip"
  mkdir -p CNRPark-Patches-150x150
  unzip -q CNRPark.zip -d CNRPark-Patches-150x150
  rm -f CNRPark.zip
else
  echo "Skip CNRPark-Patches-150x150 (already present)"
fi

# 2) CNR-EXT patches
if [ ! -d "PATCHES" ]; then
  wget -q "http://cnrpark.it/dataset/CNR-EXT-Patches-150x150.zip" -O "CNRPark_EXT.zip"
  unzip -q CNRPark_EXT.zip
  rm -f CNRPark_EXT.zip
else
  echo "Skip CNR-EXT PATCHES (already present)"
fi

# 3) PKLot
if [ ! -d "PKLot" ]; then
  wget -q "http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz" -O "PKLot.tar.gz"
  tar -xf PKLot.tar.gz
  rm -f PKLot.tar.gz
else
  echo "Skip PKLot (already present)"
fi

# 4) splits
if [ ! -d "splits" ]; then
  wget -q "http://cnrpark.it/dataset/splits.zip" -O "splits.zip"
  unzip -q splits.zip
  rm -f splits.zip
else
  echo "Skip splits (already present)"
fi

echo "Done."
