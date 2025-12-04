#!/bin/bash

# Load local Perl libraries
eval "$(perl -I$HOME/perl5/lib/perl5 -Mlocal::lib)"

datasets=("MSRpar" "MSRvid" "SMTeuroparl" "OnWN" "SMTnews")

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

    # 2. Find System Output (Updated path)
    SYS_FILE="output/STS.output.${set}.mySystem.txt"

    # Run evaluation
    if [[ "$GS_FILE" != "NOT_FOUND" && -f "$SYS_FILE" ]]; then
        echo -n "Dataset: ${set} ... "
        perl correlation.pl "$GS_FILE" "$SYS_FILE"
    else
        echo "WARNING: Could not evaluate ${set}."
        echo "  - Missing GS or SYS file in output/ folder."
    fi
done

echo "=========================================="