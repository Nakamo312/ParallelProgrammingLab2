#!/bin/bash

DEFAULT_SIZES="100 200 300 400 500"
DEFAULT_THREADS="1 2 4 8"
DEFAULT_CSV="results.csv"
DEFAULT_PLOT="performance_plot.png"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sizes)
            SIZES="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --csv)
            CSV_FILE="$2"
            shift 2
            ;;
        --plot)
            PLOT_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SIZES=${SIZES:-$DEFAULT_SIZES}
THREADS=${THREADS:-$DEFAULT_THREADS}
CSV_FILE=${CSV_FILE:-$DEFAULT_CSV}
PLOT_FILE=${PLOT_FILE:-$DEFAULT_PLOT}

if [[ ! -d venv ]]; then
    python3 -m venv venv
fi
source venv/bin/activate

python -c "import numpy" 2>/dev/null || { 
    if [[ -f requirements.txt ]]; then
        pip install -r requirements.txt
    else
        pip install numpy matplotlib pandas seaborn
    fi
}

for size in $SIZES; do
    echo "Processing size: ${size}x${size}"
    
    ./matrix_mult generate "$size" "$size" "$size" "$size" mat1.bin mat2.bin
    
    for threads in $THREADS; do
        echo -n "  Threads: $threads... "
        
        output=$(./matrix_mult multiply parallel mat1.bin mat2.bin result.bin "$threads" 2>&1)
        duration=$(echo "$output" | grep "Compute time:" | awk '{print $3}')
        gflops=$(echo "$output" | grep "GFLOPS:" | awk '{print $2}')
        
        if python validate.py mat1.bin mat2.bin result.bin \
            --csv "$CSV_FILE" \
            --duration "$duration" \
            --threads "$threads"; then
            echo "OK (${duration}ms)"
        else
            echo "Validation failed!"
        fi
        
        rm -f result.bin
    done
    
    rm -f mat1.bin mat2.bin
done

python graph.py --csv "$CSV_FILE" --output "$PLOT_FILE"

deactivate