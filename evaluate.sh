#!/bin/bash

# Load local Perl libraries
eval "$(perl -I$HOME/perl5/lib/perl5 -Mlocal::lib)"

datasets=("MSRpar" "MSRvid" "SMTeuroparl" "OnWN" "SMTnews")

# Initialize variables
total_score=0
count=0

echo "=========================================="
echo "       RUNNING MACRO EVALUATION         "
echo "=========================================="

for set in "${datasets[@]}"
do
    # 1. Find Gold Standard (GS)
    if [ -f "test-gold/STS.gs.${set}.txt" ]; then
        GS_FILE="test-gold/STS.gs.${set}.txt"
    elif [ -f "test-gold/STS.gs.surprise.${set}.txt" ]; then
        GS_FILE="test-gold/STS.gs.surprise.${set}.txt"
    else
        GS_FILE="NOT_FOUND"
    fi

    # 2. Find System Output
    SYS_FILE="output/STS.output.${set}.mySystem.txt"

    # Run evaluation
    if [[ "$GS_FILE" != "NOT_FOUND" && -f "$SYS_FILE" ]]; then
        # Capture raw output (e.g., "Pearson: 0.8345")
        RAW_OUTPUT=$(perl correlation.pl "$GS_FILE" "$SYS_FILE")
        
        # Clean output: Extract strictly the floating point number (X.XXXXX)
        SCORE=$(echo "$RAW_OUTPUT" | grep -oE '[0-9]+\.[0-9]+')
        
        # Print result to console
        echo "Dataset: ${set} ... Pearson: $SCORE"

        # Accumulate score using awk
        # If SCORE is empty (regex failed), treat as 0 to avoid crash
        if [ -z "$SCORE" ]; then SCORE=0; fi
        
        total_score=$(awk "BEGIN {print $total_score + $SCORE}")
        count=$((count + 1))
    else
        echo "WARNING: Could not evaluate ${set}."
        echo "  - Missing GS or SYS file."
    fi
done

echo "=========================================="

# Calculate and print the average
if [ $count -gt 0 ]; then
    average=$(awk "BEGIN {print $total_score / $count}")
    printf "Final Macro-Average Pearson: %.5f\n" "$average"
else
    echo "No datasets were evaluated successfully."
fi

echo "=========================================="

echo ""
echo "=========================================="
echo "        OFFICIAL POOLED EVALUATION        "
echo "=========================================="

# 1. Initialize empty combined files
> STS.gs.ALL.txt
> STS.output.ALL.txt

# 2. Concatenate all datasets into the ALL files
for set in "${datasets[@]}"
do
    # Re-locate the Gold Standard file (checking for 'surprise' variant again)
    if [ -f "test-gold/STS.gs.${set}.txt" ]; then
        GS_FILE="test-gold/STS.gs.${set}.txt"
    elif [ -f "test-gold/STS.gs.surprise.${set}.txt" ]; then
        GS_FILE="test-gold/STS.gs.surprise.${set}.txt"
    else
        GS_FILE="NOT_FOUND"
    fi

    SYS_FILE="output/STS.output.${set}.mySystem.txt"

    # If both files exist, append their contents to the ALL files
    if [[ "$GS_FILE" != "NOT_FOUND" && -f "$SYS_FILE" ]]; then
        cat "$GS_FILE" >> STS.gs.ALL.txt
        cat "$SYS_FILE" >> STS.output.ALL.txt
    fi
done

# 3. Run the official Perl script on the combined files
if [ -s STS.gs.ALL.txt ] && [ -s STS.output.ALL.txt ]; then
    RAW_OUTPUT_ALL=$(perl correlation.pl STS.gs.ALL.txt STS.output.ALL.txt)
    SCORE_ALL=$(echo "$RAW_OUTPUT_ALL" | grep -oE '[0-9]+\.[0-9]+')
    echo "Pooled ALL Pearson: $SCORE_ALL"
else
    echo "Could not calculate Pooled Average (files missing)."
fi

echo "=========================================="
# Optional: Remove temporary combined files
rm STS.gs.ALL.txt STS.output.ALL.txt