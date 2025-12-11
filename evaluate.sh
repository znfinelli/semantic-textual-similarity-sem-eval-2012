#!/bin/bash

# Load local Perl libraries
eval "$(perl -I$HOME/perl5/lib/perl5 -Mlocal::lib)"

datasets=("MSRpar" "MSRvid" "SMTeuroparl" "OnWN" "SMTnews")

# Initialize variables
total_score=0
count=0

echo "=========================================="
echo "       RUNNING SEMEVAL EVALUATION         "
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