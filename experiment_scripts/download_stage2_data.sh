#!/bin/bash

# Simple, reliable download script for stage2 data
OUTPUT_DIR="/n/netscratch/dam_lab/Lab/sqin/olmo"
MIX_FILE="src/olmo_core/data/mixes/OLMo2-7B-stage2.txt"
LOG_FILE="stage2_download.log"
BASE_URL="http://olmo-data.org"

echo "======================================"
echo "OLMo Stage2 Data Downloader"
echo "======================================"
echo "Output dir: $OUTPUT_DIR"
echo "Mix file: $MIX_FILE"
echo ""

# Count total files
total=$(grep -v '^#' "$MIX_FILE" | grep -v '^[[:space:]]*$' | wc -l)
echo "Total files to download: $total"
echo ""

# Initialize counters
downloaded=0
skipped=0
failed=0

# Read mix file and download each file
while IFS=',' read -r prefix path; do
    # Skip comments and empty lines
    [[ "$prefix" =~ ^#.*$ ]] && continue
    [[ -z "$prefix" ]] && continue

    # Handle lines without comma (just path)
    if [ -z "$path" ]; then
        path="$prefix"
        prefix=""
    fi

    # Construct output path
    if [ -n "$prefix" ]; then
        output_path="${OUTPUT_DIR}/${prefix}/${path}"
    else
        output_path="${OUTPUT_DIR}/${path}"
    fi

    # Check if file already exists and is non-empty
    if [ -f "$output_path" ] && [ -s "$output_path" ]; then
        size=$(stat -f%z "$output_path" 2>/dev/null || stat -c%s "$output_path" 2>/dev/null)
        if [ "$size" -gt 10000 ]; then
            echo "SKIP: $(basename $output_path) (already exists, ${size} bytes)"
            ((skipped++))
            continue
        else
            # File exists but is too small, redownload (remove it first to avoid partial resume)
            echo "REDOWNLOAD: $(basename $output_path) (existing file too small: ${size} bytes)"
            rm -f "$output_path"
        fi
    fi

    # Create directory
    mkdir -p "$(dirname "$output_path")"

    # Download with continue support for interrupted downloads
    # Use temporary file to avoid corrupting existing partial downloads
    url="${BASE_URL}/${path}"
    temp_path="${output_path}.tmp"

    echo "DOWNLOADING: $(basename $output_path)..."

    # Remove any old temp file
    rm -f "$temp_path"

    if wget -q --timeout=60 --tries=3 -O "$temp_path" "$url"; then
        # Move temp file to final location only if successful
        mv "$temp_path" "$output_path"
        # Verify download
        if [ -f "$output_path" ] && [ -s "$output_path" ]; then
            size=$(stat -f%z "$output_path" 2>/dev/null || stat -c%s "$output_path" 2>/dev/null)
            if [ "$size" -gt 10000 ]; then
                echo "  ✓ Success (${size} bytes)"
                ((downloaded++))
            else
                echo "  ✗ Failed (file too small: ${size} bytes, likely 404 error)"
                rm -f "$output_path"
                ((failed++))
            fi
        else
            echo "  ✗ Failed (file not created)"
            ((failed++))
        fi
    else
        echo "  ✗ Failed (wget error)"
        rm -f "$output_path" "$temp_path"
        ((failed++))
    fi

done < <(grep -v '^#' "$MIX_FILE" | grep -v '^[[:space:]]*$')

echo ""
echo "======================================"
echo "Download Complete!"
echo "======================================"
echo "Total files:     $total"
echo "Downloaded:      $downloaded"
echo "Skipped:         $skipped"
echo "Failed:          $failed"
echo "======================================"

if [ "$failed" -gt 0 ]; then
    echo ""
    echo "⚠ Warning: $failed files failed to download."
    echo "These may not exist on the server or paths may be incorrect."
fi
