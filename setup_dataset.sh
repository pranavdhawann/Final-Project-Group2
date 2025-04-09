#!/bin/bash

# === Configuration ===
KAGGLE_JSON="kaggle.json"
KAGGLE_DIR="$HOME/.kaggle"
COMPETITION="byu-locating-bacterial-flagellar-motors-2025"
DEST_DIR="$(pwd)/Dataset"
mkdir DESt_DIR
echo "🚀 Starting Kaggle dataset download setup..."

# === Step 1: Validate kaggle.json ===
if [ -f "$KAGGLE_JSON" ]; then
    echo "📄 Found $KAGGLE_JSON in current directory. Moving it to $KAGGLE_DIR..."
    mkdir -p "$KAGGLE_DIR"
    mv "$KAGGLE_JSON" "$KAGGLE_DIR/"
    chmod 600 "$KAGGLE_DIR/kaggle.json"
elif [ -f "$KAGGLE_DIR/kaggle.json" ]; then
    echo "✅ Found existing kaggle.json in $KAGGLE_DIR. Skipping move."
else
    echo "❌ Error: kaggle.json not found in current directory or $KAGGLE_DIR."
    echo "👉 Please upload your kaggle.json API file first."
    exit 1
fi

# === Step 2: Set KAGGLE_CONFIG_DIR for kagglehub ===
export KAGGLE_CONFIG_DIR="$KAGGLE_DIR"
echo "🔐 KAGGLE_CONFIG_DIR set to $KAGGLE_CONFIG_DIR"

# === Step 3: Create destination directory ===
mkdir -p "$DEST_DIR"
echo "📁 Destination directory is $DEST_DIR"

# === Step 4: Run embedded Python to download dataset ===
echo "🐍 Running Python code to download and move the dataset..."
pip install kagglehub
python3 -u - <<EOF
import kagglehub
import shutil
import os

print("📥 Downloading dataset from KaggleHub...", flush=True)
path = kagglehub.competition_download("$COMPETITION")
print(f"📦 Dataset downloaded to cache directory: {path}", flush=True)

# Define destination directory
dest = "$DEST_DIR"

# Move contents to the destination directory
for item in os.listdir(path):
    src = os.path.join(path, item)
    dst = os.path.join(dest, item)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"📁 Copied directory: {item}", flush=True)
    else:
        shutil.copy2(src, dst)
        print(f"📄 Copied file: {item}", flush=True)

print("✅ Dataset successfully copied to Dataset/ folder.", flush=True)
EOF

echo "🎉 All done! Dataset is in ./Dataset/"
