#!/bin/bash

################################################################################
# Quick Test Script for Benchmark Automation
# This script tests individual components without running full benchmarks
################################################################################

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "========================================"
echo "Testing Benchmark Automation Components"
echo "========================================"
echo ""

# Test 1: Check system info
echo "Test 1: System Information"
echo "  Architecture: $(uname -m)"
echo "  CPU cores: $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 'unknown')"
echo "  Python: $(python --version 2>&1 || python3 --version 2>&1)"
if command -v cmake &> /dev/null; then
    echo -e "  CMake: ${GREEN}✓${NC} $(cmake --version | head -1)"
else
    echo -e "  CMake: ${RED}✗ Not found${NC}"
fi
if command -v clang &> /dev/null; then
    echo -e "  Clang: ${GREEN}✓${NC} $(clang --version | head -1)"
else
    echo -e "  Clang: ${RED}✗ Not found${NC}"
fi
echo ""

# Test 2: Check required files
echo "Test 2: Required Files"
files=(
    "embed_quant.sh"
    "tune_gemm_blocks.sh"
    "utils/convert-helper-bitnet.py"
    "requirements.txt"
)
for f in "${files[@]}"; do
    if [[ -f "$f" ]]; then
        echo -e "  $f: ${GREEN}✓${NC}"
    else
        echo -e "  $f: ${RED}✗ Missing${NC}"
    fi
done
echo ""

# Test 3: Check build directory
echo "Test 3: Build Status"
if [[ -d "build" ]]; then
    echo -e "  build/ directory: ${GREEN}✓${NC}"
    if [[ -f "build/bin/llama-bench" ]]; then
        echo -e "  llama-bench: ${GREEN}✓${NC}"
    else
        echo -e "  llama-bench: ${RED}✗ Not built${NC}"
    fi
    if [[ -f "build/bin/llama-perplexity" ]]; then
        echo -e "  llama-perplexity: ${GREEN}✓${NC}"
    else
        echo -e "  llama-perplexity: ${RED}✗ Not built${NC}"
    fi
    if [[ -f "build/bin/llama-quantize" ]]; then
        echo -e "  llama-quantize: ${GREEN}✓${NC}"
    else
        echo -e "  llama-quantize: ${RED}✗ Not built${NC}"
    fi
else
    echo -e "  build/ directory: ${RED}✗ Not found${NC}"
fi
echo ""

# Test 4: Check data directory
echo "Test 4: Benchmark Datasets"
datasets=(
    "data/wikitext-2-raw/wiki.test.raw"
    "data/ptb/ptb.test.txt"
    "data/lambada/lambada_test_plain_text.txt"
    "data/clue/tnews.test.txt"
)
found=0
for ds in "${datasets[@]}"; do
    if [[ -f "$ds" ]]; then
        echo -e "  $(basename $(dirname $ds)): ${GREEN}✓${NC}"
        found=$((found + 1))
    else
        echo -e "  $(basename $(dirname $ds)): ${RED}✗ Not found${NC}"
    fi
done
echo "  Total: $found/4 datasets available"
echo ""

# Test 5: Check models
echo "Test 5: Model Files"
MODEL_DIR="models/BitNet-b1.58-2B-4T"
if [[ -d "$MODEL_DIR" ]]; then
    echo -e "  Model directory: ${GREEN}✓${NC}"
    if [[ -f "$MODEL_DIR/ggml-model-f32.gguf" ]]; then
        echo -e "  F32 model: ${GREEN}✓${NC}"
    else
        echo -e "  F32 model: ${RED}✗ Not found${NC}"
    fi
    
    # Count quantized models
    quant_count=$(ls "$MODEL_DIR"/ggml-model-i2_s_embed_*.gguf 2>/dev/null | wc -l)
    if [[ $quant_count -gt 0 ]]; then
        echo -e "  Quantized models: ${GREEN}✓${NC} ($quant_count files)"
    else
        echo -e "  Quantized models: ${RED}✗ None found${NC}"
    fi
else
    echo -e "  Model directory: ${RED}✗ Not found${NC}"
fi
echo ""

# Test 6: Thread count generation
echo "Test 6: Thread Configuration"
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "8")
THREAD_COUNTS="1"
for ((i=2; i<=NPROC; i*=2)); do
    THREAD_COUNTS="${THREAD_COUNTS},${i}"
done
echo "  Max threads: $NPROC"
echo "  Test thread counts: $THREAD_COUNTS"
echo ""

# Test 7: Check stats directory
echo "Test 7: Output Directory"
if [[ -d "stats" ]]; then
    echo -e "  stats/ directory: ${GREEN}✓${NC}"
    file_count=$(ls stats/ 2>/dev/null | wc -l)
    echo "  Files in stats/: $file_count"
else
    echo -e "  stats/ directory: ${RED}✗ Not found${NC}"
    echo "  Creating stats/ directory..."
    mkdir -p stats
    echo -e "  ${GREEN}✓ Created${NC}"
fi
echo ""

# Summary
echo "========================================"
echo "Test Summary"
echo "========================================"
echo ""
echo "To run the full benchmark automation:"
echo "  ./run_paper_benchmarks.sh"
echo ""
echo "To build the project first (if not built):"
echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release"
echo "  cmake --build build --config Release"
echo ""
echo "To download and convert the model:"
echo "  huggingface-cli download microsoft/BitNet-b1.58-2B-4T --local-dir models/BitNet-b1.58-2B-4T"
echo "  python utils/convert-helper-bitnet.py models/BitNet-b1.58-2B-4T"
echo ""
