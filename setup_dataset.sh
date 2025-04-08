#!/bin/bash
pip install kaggle

# Define constants
KAGGLE_DIR="$HOME/.kaggle"
KAGGLE_JSON="kaggle.json"
COMPETITION="byu-locating-bacterial-flagellar-motors-2025"
ZIP_FILE="$COMPETITION.zip"
EXTRACT_DIR="$COMPETITION"

# Step 1: Check if kaggle.json exists in current directory
if [ -f "$KAGGLE_JSON" ]; then
    echo "📄 Found $KAGGLE_JSON in current directory. Moving it to $KAGGLE_DIR..."
    mkdir -p "$KAGGLE_DIR"
    mv "$KAGGLE_JSON" "$KAGGLE_DIR/"
    chmod 600 "$KAGGLE_DIR/$KAGGLE_JSON"
elif [ -f "$KAGGLE_DIR/$KAGGLE_JSON" ]; then
    echo "📄 Found existing $KAGGLE_JSON in $KAGGLE_DIR. Skipping move."
else
    echo "❌ Error: $KAGGLE_JSON not found in current directory or $KAGGLE_DIR"
    echo "👉 Please upload your kaggle.json API file to this directory first."
    exit 1
fi

# Step 2: Create .kaggle directory if it doesn't exist
echo "📁 Creating Kaggle config directory at $KAGGLE_DIR..."
mkdir -p "$KAGGLE_DIR"

# Step 3: Move kaggle.json and set permissions
echo "🔐 Setting up Kaggle API credentials..."
mv "$KAGGLE_JSON" "$KAGGLE_DIR/"
chmod 600 "$KAGGLE_DIR/$KAGGLE_JSON"

# Step 4: Download the competition dataset
echo "⬇️ Downloading dataset for competition: $COMPETITION..."
kaggle competitions download -c "$COMPETITION"

# Check if download was successful
if [ ! -f "$ZIP_FILE" ]; then
    echo "❌ Error: Failed to download the dataset. Please check competition name or kaggle setup."
    exit 1
fi

# Step 5: Unzip the dataset
echo "📦 Unzipping $ZIP_FILE into $EXTRACT_DIR/..."
mkdir -p "$EXTRACT_DIR"
unzip -q "$ZIP_FILE" -d "$EXTRACT_DIR/"

# Step 6: Verify unzip success
if [ $? -eq 0 ]; then
    echo "✅ Dataset downloaded and extracted to $EXTRACT_DIR/"
else
    echo "⚠️ Unzip failed. Please check if the zip file is valid."
    exit 1
fi

