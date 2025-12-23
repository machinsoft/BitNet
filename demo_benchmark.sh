#!/bin/bash

################################################################################
# Quick Demo of Benchmark Automation
# This runs a subset of benchmarks to verify the script works
################################################################################

set -euo pipefail

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

STATS_DIR="stats/demo_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${STATS_DIR}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Quick Benchmark Demo${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Output directory: ${STATS_DIR}"
echo ""

# Test 1: Machine info
echo -e "${GREEN}[1/3] Collecting machine info...${NC}"
{
    echo "=== Machine Information ==="
    echo "Architecture: $(uname -m)"
    echo "CPU cores: $(nproc)"
    echo "Timestamp: $(date)"
    echo ""
    lscpu | head -20
} | tee "${STATS_DIR}/machine_info.txt"
echo ""

# Test 2: Quick benchmark test
echo -e "${GREEN}[2/3] Running quick benchmark (2 threads only)...${NC}"
if [[ -f "build/bin/llama-bench" ]] && [[ -f "models/BitNet-b1.58-2B-4T/ggml-model-i2_s_embed_q6_k.gguf" ]]; then
    ./build/bin/llama-bench \
        -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s_embed_q6_k.gguf \
        -p 128 -n 128 -t 1,2,4 -ngl 0 \
        2>&1 | tee "${STATS_DIR}/bench_quick.txt"
    
    # Parse results
    {
        echo "# Quick Benchmark Results"
        echo ""
        echo "| Threads | Test | Tokens/sec |"
        echo "|---------|------|------------|"
        
        awk -F '|' '
            /bitnet.*pp128/ || /bitnet.*tg128/ {
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", $6);
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", $7);
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", $8);
                split($8, perf, "±");
                printf "| %7s | %4s | %10s |\n", $6, $7, perf[1];
            }
        ' "${STATS_DIR}/bench_quick.txt"
    } > "${STATS_DIR}/bench_results.md"
    
    echo ""
    echo -e "${GREEN}Results saved to: ${STATS_DIR}/bench_results.md${NC}"
    cat "${STATS_DIR}/bench_results.md"
else
    echo "Skipping benchmark (model or binary not found)"
fi
echo ""

# Test 3: Quick PPL test (one dataset only)
echo -e "${GREEN}[3/3] Running quick PPL test (wikitext-2 only, 2 embed types)...${NC}"
if [[ -f "build/bin/llama-perplexity" ]] && [[ -f "data/wikitext-2-raw/wiki.test.raw" ]]; then
    {
        echo "# Quick PPL Test"
        echo ""
        echo "| Embed Type | PPL |"
        echo "|------------|-----|"
        
        for embed in i2_s q6_k; do
            model="models/BitNet-b1.58-2B-4T/ggml-model-i2_s_embed_${embed}.gguf"
            if [[ -f "$model" ]]; then
                echo "Testing: $embed..."
                output=$(./build/bin/llama-perplexity \
                    -m "$model" \
                    -f data/wikitext-2-raw/wiki.test.raw \
                    -t 4 -ngl 0 2>&1 || true)
                
                ppl=$(echo "$output" | awk '
                    /Final estimate/ && /PPL/ {
                        if (match($0, /PPL[[:space:]]*=[[:space:]]*([0-9]+(\.[0-9]+)?)/, m)) {
                            print m[1];
                            exit;
                        }
                    }
                ')
                
                if [[ -n "$ppl" ]]; then
                    echo "| $embed | $ppl |"
                else
                    echo "| $embed | N/A |"
                fi
            fi
        done
    } | tee "${STATS_DIR}/ppl_quick.md"
    
    echo ""
    echo -e "${GREEN}Results saved to: ${STATS_DIR}/ppl_quick.md${NC}"
else
    echo "Skipping PPL test (binary or dataset not found)"
fi
echo ""

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Demo completed!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "All results in: ${STATS_DIR}/"
echo ""
echo "To run the full automation script:"
echo "  ./run_paper_benchmarks.sh"
echo ""
