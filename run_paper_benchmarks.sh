#!/bin/bash

################################################################################
# Paper Benchmark Automation Script
# This script automates all experiments needed for the paper on both Intel and ARM
################################################################################

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
STATS_DIR="stats"
MODEL_NAME="BitNet-b1.58-2B-4T"
MODEL_DIR="models/${MODEL_NAME}"
HF_REPO="microsoft/${MODEL_NAME}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MACHINE_INFO_FILE="${STATS_DIR}/machine_info_${TIMESTAMP}.txt"
BENCH_RESULTS_FILE="${STATS_DIR}/bench_results_${TIMESTAMP}.md"
BENCH_RAW_FILE="${STATS_DIR}/bench_raw_${TIMESTAMP}.txt"
PPL_RESULTS_FILE="${STATS_DIR}/ppl_results_${TIMESTAMP}.md"
PPL_CSV_FILE="${STATS_DIR}/ppl_results_${TIMESTAMP}.csv"

# Create stats directory if not exists
mkdir -p "${STATS_DIR}"

################################################################################
# Helper Functions
################################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

section_header() {
    echo ""
    echo "================================================================================"
    echo -e "${GREEN}$1${NC}"
    echo "================================================================================"
}

################################################################################
# Step 1: Machine Information and Environment Setup
################################################################################

step1_machine_info() {
    section_header "STEP 1: Machine Information and Environment Setup"
    
    log_info "Collecting machine information..."
    
    {
        echo "================================"
        echo "Machine Information"
        echo "================================"
        echo "Timestamp: $(date)"
        echo ""
        
        echo "--- System Architecture ---"
        uname -a
        echo ""
        
        echo "--- CPU Information ---"
        if command -v lscpu &> /dev/null; then
            lscpu
        elif [[ -f /proc/cpuinfo ]]; then
            cat /proc/cpuinfo
        else
            log_warning "Could not get CPU information"
        fi
        echo ""
        
        echo "--- CPU Cores ---"
        NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "unknown")
        echo "Number of CPU cores: ${NPROC}"
        echo ""
        
        echo "--- Memory Information ---"
        if command -v free &> /dev/null; then
            free -h
        elif command -v vm_stat &> /dev/null; then
            vm_stat
        else
            log_warning "Could not get memory information"
        fi
        echo ""
        
        echo "--- Architecture Detection ---"
        ARCH=$(uname -m)
        echo "Architecture: ${ARCH}"
        if [[ "${ARCH}" == "x86_64" ]]; then
            echo "Platform: Intel/AMD x86_64"
        elif [[ "${ARCH}" == "aarch64" ]] || [[ "${ARCH}" == "arm64" ]]; then
            echo "Platform: ARM64"
        else
            echo "Platform: Other (${ARCH})"
        fi
        echo ""
        
        echo "--- Compiler Information ---"
        if command -v clang &> /dev/null; then
            clang --version
        fi
        if command -v gcc &> /dev/null; then
            gcc --version
        fi
        if command -v cmake &> /dev/null; then
            cmake --version
        fi
        echo ""
        
        echo "--- Python Environment ---"
        python --version || python3 --version
        if command -v conda &> /dev/null; then
            conda --version
            echo "Active conda environment: ${CONDA_DEFAULT_ENV:-none}"
        fi
        echo ""
        
    } | tee "${MACHINE_INFO_FILE}"
    
    log_success "Machine information saved to: ${MACHINE_INFO_FILE}"
    
    # Install dependencies according to README
    log_info "Installing Python dependencies..."
    if [[ -f requirements.txt ]]; then
        pip install -r requirements.txt
        log_success "Python dependencies installed"
    else
        log_warning "requirements.txt not found, skipping dependency installation"
    fi
}

################################################################################
# Step 2: Build Project
################################################################################

step2_build() {
    section_header "STEP 2: Building Project"
    
    log_info "Configuring CMake..."
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    
    log_info "Building project..."
    cmake --build build --config Release
    
    log_success "Build completed successfully"
}

################################################################################
# Step 3: Download and Convert Model
################################################################################

step3_download_convert() {
    section_header "STEP 3: Download and Convert Model"
    
    if [[ -d "${MODEL_DIR}" ]] && [[ -f "${MODEL_DIR}/ggml-model-f32.gguf" ]]; then
        log_warning "Model directory already exists and contains f32 model, skipping download"
        read -p "Do you want to re-download and convert? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return
        fi
    fi
    
    # Create model directory
    mkdir -p "${MODEL_DIR}"
    
    # Download from HuggingFace
    log_info "Downloading model from HuggingFace: ${HF_REPO}"
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download "${HF_REPO}" --local-dir "${MODEL_DIR}"
    else
        log_error "huggingface-cli not found. Please install it with: pip install huggingface_hub"
        exit 1
    fi
    
    # Convert to f32 GGUF using the helper script
    log_info "Converting model to f32 GGUF format..."
    if [[ -f "utils/convert-helper-bitnet.py" ]]; then
        # The script creates ggml-model-f32-bitnet.gguf, we'll rename it
        python utils/convert-helper-bitnet.py "${MODEL_DIR}"
        
        # Rename the output to match expected name
        if [[ -f "${MODEL_DIR}/ggml-model-f32-bitnet.gguf" ]]; then
            mv "${MODEL_DIR}/ggml-model-f32-bitnet.gguf" "${MODEL_DIR}/ggml-model-f32.gguf"
        fi
    else
        log_error "Convert helper script not found"
        exit 1
    fi
    
    log_success "Model downloaded and converted to f32 GGUF"
}

################################################################################
# Step 4: Quantize Embeddings
################################################################################

step4_quantize_embeddings() {
    section_header "STEP 4: Quantize Embeddings"
    
    log_info "Running embed_quant.sh to create different embedding quantization variants..."
    
    if [[ ! -f "embed_quant.sh" ]]; then
        log_error "embed_quant.sh not found"
        exit 1
    fi
    
    bash embed_quant.sh
    
    log_success "Embedding quantization completed"
}

################################################################################
# Step 5: Tune GEMM Block Sizes
################################################################################

step5_tune_gemm() {
    section_header "STEP 5: Tune GEMM Block Sizes"
    
    log_info "Running GEMM block size tuning..."
    
    # Backup original tune script if needed
    if [[ ! -f "tune_gemm_blocks.sh.bak" ]]; then
        cp tune_gemm_blocks.sh tune_gemm_blocks.sh.bak
    fi
    
    # Get number of threads
    NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "8")
    
    # Update the tuning script to use a broader search space
    log_info "Updating tune_gemm_blocks.sh for comprehensive search..."
    
    # Create a temporary tuning script with broader search
    cat > tune_gemm_blocks_auto.sh << 'EOF'
#!/bin/bash
set -e

HEADER_FILE="include/gemm-config.h"
BENCH_CMD="./build/bin/llama-bench -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s_embed_i2_s.gguf -p 128 -n 0 -t 16 -ngl 0"
BUILD_CMD="cmake --build build --config Release -j"

ACT_PARALLEL_DEFINE=true

# Expanded search space for better tuning
ROW_BLOCK_VALUES=(2 4 8)
COL_BLOCK_VALUES=(64 128 256)
PARALLEL_SIZE_VALUES=(2 4 8)

BEST_PERF=0
BEST_ROW_BLOCK=0
BEST_COL_BLOCK=0
BEST_PARALLEL_SIZE=0
LOG_FILE="stats/tuning_log.csv"

if [ -f "$HEADER_FILE" ]; then
    cp "$HEADER_FILE" "${HEADER_FILE}.bak"
fi

echo "Starting comprehensive tuning process..."
echo "row_block,col_block,parallel_size,tokens_per_second" > "$LOG_FILE"

cleanup() {
    echo "Restoring original header file..."
    if [ -f "${HEADER_FILE}.bak" ]; then
        mv "${HEADER_FILE}.bak" "$HEADER_FILE"
    fi
    echo "Tuning finished."
    echo "Best: ROW_BLOCK=${BEST_ROW_BLOCK}, COL_BLOCK=${BEST_COL_BLOCK}, PARALLEL=${BEST_PARALLEL_SIZE} -> ${BEST_PERF} tokens/s"
}

trap cleanup EXIT

for ps in "${PARALLEL_SIZE_VALUES[@]}"; do
    for rb in "${ROW_BLOCK_VALUES[@]}"; do
        for cb in "${COL_BLOCK_VALUES[@]}"; do
            echo "Testing: ROW=${rb}, COL=${cb}, PARALLEL=${ps}"
            
            echo "// Auto-generated by tuning script" > "$HEADER_FILE"
            if [ "$ACT_PARALLEL_DEFINE" = "true" ]; then
                echo "#define ACT_PARALLEL" >> "$HEADER_FILE"
            fi
            echo "#if defined(ACT_PARALLEL)" >> "$HEADER_FILE"
            echo "    #define ROW_BLOCK_SIZE ${rb}" >> "$HEADER_FILE"
            echo "    #define COL_BLOCK_SIZE ${cb}" >> "$HEADER_FILE"
            echo "    #define PARALLEL_SIZE ${ps}" >> "$HEADER_FILE"
            echo "#else" >> "$HEADER_FILE"
            echo "    #define ROW_BLOCK_SIZE ${rb}" >> "$HEADER_FILE"
            echo "    #define COL_BLOCK_SIZE ${cb}" >> "$HEADER_FILE"
            echo "    #define PARALLEL_SIZE ${ps}" >> "$HEADER_FILE"
            echo "#endif" >> "$HEADER_FILE"
            
            $BUILD_CMD > /dev/null 2>&1
            
            output=$(eval "$BENCH_CMD" 2>&1)
            
            perf=$(echo "$output" | awk -F '|' '
                /pp128/ && /bitnet/ {
                    gsub(/ /, "", $8);
                    split($8, perf, "±");
                    print perf[1];
                    exit;
                }
            ')
            
            if [ -z "$perf" ]; then
                perf=0
            fi
            
            echo "Performance: ${perf} tokens/s"
            echo "${rb},${cb},${ps},${perf}" >> "$LOG_FILE"
            
            if (( $(echo "$perf > $BEST_PERF" | bc -l) )); then
                BEST_PERF=$perf
                BEST_ROW_BLOCK=$rb
                BEST_COL_BLOCK=$cb
                BEST_PARALLEL_SIZE=$ps
                echo "*** New best found! ***"
            fi
        done
    done
done

echo "Best configuration: ROW=${BEST_ROW_BLOCK}, COL=${BEST_COL_BLOCK}, PARALLEL=${BEST_PARALLEL_SIZE}"
echo "Best performance: ${BEST_PERF} tokens/s"
EOF
    
    chmod +x tune_gemm_blocks_auto.sh
    bash tune_gemm_blocks_auto.sh
    
    # Read the best configuration from the log
    if [[ -f "stats/tuning_log.csv" ]]; then
        BEST_CONFIG=$(tail -n +2 "stats/tuning_log.csv" | sort -t',' -k4 -nr | head -1)
        BEST_ROW=$(echo "$BEST_CONFIG" | cut -d',' -f1)
        BEST_COL=$(echo "$BEST_CONFIG" | cut -d',' -f2)
        BEST_PAR=$(echo "$BEST_CONFIG" | cut -d',' -f3)
        BEST_PERF=$(echo "$BEST_CONFIG" | cut -d',' -f4)
        
        log_success "Best configuration found:"
        log_success "  ROW_BLOCK_SIZE=${BEST_ROW}, COL_BLOCK_SIZE=${BEST_COL}, PARALLEL_SIZE=${BEST_PAR}"
        log_success "  Performance: ${BEST_PERF} tokens/s"
        
        # Apply the best configuration
        log_info "Applying best configuration to gemm-config.h..."
        cat > include/gemm-config.h << EOF
// Auto-generated with best tuning results
// Best performance: ${BEST_PERF} tokens/s
#define ACT_PARALLEL
#if defined(ACT_PARALLEL)
    #define ROW_BLOCK_SIZE ${BEST_ROW}
    #define COL_BLOCK_SIZE ${BEST_COL}
    #define PARALLEL_SIZE ${BEST_PAR}
#else
    #define ROW_BLOCK_SIZE ${BEST_ROW}
    #define COL_BLOCK_SIZE ${BEST_COL}
    #define PARALLEL_SIZE ${BEST_PAR}
#endif
EOF
        
        # Rebuild with best configuration
        log_info "Rebuilding with best configuration..."
        cmake --build build --config Release -j
        
        log_success "GEMM tuning completed and applied"
    else
        log_error "Tuning log not found"
    fi
}

################################################################################
# Step 6: Run Performance Benchmarks
################################################################################

step6_benchmark() {
    section_header "STEP 6: Running Performance Benchmarks"
    
    # Get number of threads for this machine
    NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "8")
    log_info "Detected ${NPROC} CPU cores"
    
    # Generate thread counts: 1, 2, 4, 8, 16, ...
    THREAD_COUNTS="1"
    for ((i=2; i<=NPROC; i*=2)); do
        THREAD_COUNTS="${THREAD_COUNTS},${i}"
    done
    
    log_info "Testing with thread counts: ${THREAD_COUNTS}"
    
    # Create benchmark script
    cat > bench.sh << EOF
#!/bin/bash
set -e

MODEL="${MODEL_DIR}/ggml-model-i2_s_embed_q6_k.gguf"
THREADS="${THREAD_COUNTS}"

if [[ ! -f "\${MODEL}" ]]; then
    echo "Error: Model not found: \${MODEL}"
    exit 1
fi

./build/bin/llama-bench -m "\${MODEL}" -p 128 -n 128 -t "\${THREADS}" -ngl 0
EOF
    
    chmod +x bench.sh
    
    log_info "Running benchmark..."
    
    # Run benchmark and capture output
    ./bench.sh 2>&1 | tee "${BENCH_RAW_FILE}"
    
    # Parse and format results
    log_info "Parsing benchmark results..."
    
    {
        echo "# Benchmark Results"
        echo ""
        echo "**Machine:** $(uname -m)"
        echo "**Timestamp:** $(date)"
        echo "**Model:** ${MODEL_NAME}"
        echo "**Quantization:** I2_S weight, Q6_K embeddings"
        echo ""
        echo "## Performance Summary"
        echo ""
        echo "| Threads | Test Type | Tokens/sec | Std Dev |"
        echo "|---------|-----------|------------|---------|"
        
        awk -F '|' '
            /bitnet.*pp128/ || /bitnet.*tg128/ {
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", $6);  # threads
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", $7);  # test
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", $8);  # t/s
                
                threads = $6;
                test = $7;
                
                split($8, perf, "±");
                tokens = perf[1];
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", tokens);
                
                stddev = perf[2];
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", stddev);
                
                printf "| %7s | %9s | %10s | %7s |\n", threads, test, tokens, stddev;
            }
        ' "${BENCH_RAW_FILE}"
        
        echo ""
        echo "## Detailed Output"
        echo ""
        echo '```'
        cat "${BENCH_RAW_FILE}"
        echo '```'
        
    } > "${BENCH_RESULTS_FILE}"
    
    log_success "Benchmark results saved to: ${BENCH_RESULTS_FILE}"
}

################################################################################
# Step 7: Run PPL Benchmarks
################################################################################

step7_ppl_benchmark() {
    section_header "STEP 7: Running Perplexity (PPL) Benchmarks"
    
    log_info "Checking benchmark datasets..."
    
    # Check which datasets are available
    DATASETS=""
    for ds in data/wikitext-2-raw/wiki.test.raw data/ptb/ptb.test.txt data/lambada/lambada_test_plain_text.txt data/clue/tnews.test.txt; do
        if [[ -f "$ds" ]]; then
            DATASETS="${DATASETS} ${ds}"
            log_info "Found dataset: ${ds}"
        else
            log_warning "Dataset not found: ${ds}"
        fi
    done
    
    if [[ -z "${DATASETS}" ]]; then
        log_error "No benchmark datasets found in data/ directory"
        log_warning "Skipping PPL benchmarks"
        return
    fi
    
    log_info "Creating PPL benchmark script..."
    
    # Create a modified PPL script
    cat > embed_quant_ppl_auto.sh << 'EOFPPL'
#!/usr/bin/env bash
set -euo pipefail

BIN="./build/bin/llama-perplexity"
MODEL_DIR="models/BitNet-b1.58-2B-4T"
MODEL_TEMPLATE="ggml-model-i2_s_embed_{ET}.gguf"

EMBED_TYPES="f32 bf16 f16 i2_s q3_k q4_0 q5_0 q6_k tq1_0 tq2_0"
DATASETS="DATASETS_PLACEHOLDER"

THREADS="${THREADS:-16}"
NGL="${NGL:-0}"

CSV_LOG="ppl_results_temp.csv"

if [[ ! -x "$BIN" ]]; then
  echo "Error: llama-perplexity not found at $BIN" >&2
  exit 1
fi

model_size_mib() {
  local f="$1"
  local sz
  sz=$(stat -c %s "$f" 2>/dev/null || stat -f %z "$f" 2>/dev/null || echo 0)
  awk -v b="$sz" 'BEGIN { printf("%.2f", b/1024/1024) }'
}

extract_ppl_final() {
  awk '
    /Final estimate/ && /PPL/ {
      if (match($0, /PPL[[:space:]]*=[[:space:]]*([0-9]+(\.[0-9]+)?)\s*\+\/\-\s*([0-9]+(\.[0-9]+)?)/, m)) {
        print m[1] "," m[3];
        found=1;
      }
    }
    END { if (!found) exit 1 }
  '
}

extract_perplexity() {
  awk '
    {
      for (i=1; i<=NF; ++i) {
        if (tolower($i) ~ /perplexity/) {
          for (j=i; j<=NF; ++j) {
            if ($j ~ /^[0-9]+(\.[0-9]+)?$/) { p=$j; break }
            gsub(/^.*=/, "", $j); gsub(/,$/, "", $j); gsub(/^\(/, "", $j); gsub(/\)$/, "", $j)
            if ($j ~ /^[0-9]+(\.[0-9]+)?$/) { p=$j; break }
          }
        }
      }
      if (p) last=p
    }
    END { if (last) print last }'
}

echo "| embed-type |           model |   size | dataset | threads |        ppl |"
echo "| ---------- | --------------: | -----: | ------: | ------: | ---------: |"
echo "embed_type,model,model_size_mib,dataset,threads,perplexity,perplexity_err" > "$CSV_LOG"

for et in $EMBED_TYPES; do
  model_glob="${MODEL_DIR}/$(echo "$MODEL_TEMPLATE" | sed "s/{ET}/$et/")"
  
  found_any=0
  for model in $model_glob; do
    [[ -e "$model" ]] || continue
    found_any=1
  done
  
  if [[ $found_any -eq 0 ]]; then
    echo "Warning: no models found for embed type '$et', skipping." >&2
    continue
  fi

  for model in $model_glob; do
    [[ -e "$model" ]] || continue
    size_mib=$(model_size_mib "$model")

    for ds in $DATASETS; do
      if [[ ! -r "$ds" ]]; then
        echo "Warning: dataset not found: $ds (skipping)" >&2
        continue
      fi

      echo "==> Testing: model=$model, dataset=$ds"
      out=$("$BIN" -m "$model" -f "$ds" -t "$THREADS" -ngl "$NGL" 2>&1 || true)

      ppl_pair=$(echo "$out" | extract_ppl_final || true)
      if [[ -n "${ppl_pair:-}" ]]; then
        ppl="${ppl_pair%%,*}"
        ppl_err="${ppl_pair##*,}"
      else
        ppl=$(echo "$out" | extract_perplexity || true)
        if [[ -z "${ppl:-}" ]]; then
          ppl="NA"
        fi
        ppl_err="NA"
      fi

      if [[ "$ppl_err" != "NA" ]]; then
        ppl_disp="$ppl ± $ppl_err"
      else
        ppl_disp="$ppl"
      fi

      printf "| %10s | %14s | %6s MiB | %7s | %7s | %10s |\n" \
        "$et" "$(basename "$model")" "$size_mib" "$(basename "$ds")" "$THREADS" "$ppl_disp"

      echo "$et,$(basename "$model"),$size_mib,$(basename "$ds"),$THREADS,$ppl,$ppl_err" >> "$CSV_LOG"
    done
  done
done

echo "Done. Results saved to $CSV_LOG"
EOFPPL
    
    # Replace DATASETS placeholder
    sed -i "s|DATASETS_PLACEHOLDER|${DATASETS}|g" embed_quant_ppl_auto.sh
    chmod +x embed_quant_ppl_auto.sh
    
    log_info "Running PPL benchmarks (this may take a while)..."
    
    # Run the PPL benchmark
    ./embed_quant_ppl_auto.sh 2>&1 | tee "${PPL_RESULTS_FILE}.raw"
    
    # Format the results
    {
        echo "# Perplexity (PPL) Benchmark Results"
        echo ""
        echo "**Machine:** $(uname -m)"
        echo "**Timestamp:** $(date)"
        echo "**Model:** ${MODEL_NAME}"
        echo ""
        echo "## Results by Embedding Type"
        echo ""
        
        grep "^|" "${PPL_RESULTS_FILE}.raw" || true
        
        echo ""
        echo "## Summary Statistics"
        echo ""
        
        if [[ -f "ppl_results_temp.csv" ]]; then
            # Copy to final location
            cp ppl_results_temp.csv "${PPL_CSV_FILE}"
            
            # Generate summary by embed type
            echo "### Average PPL by Embedding Type"
            echo ""
            echo "| Embed Type | Avg PPL | Models Tested |"
            echo "|------------|---------|---------------|"
            
            awk -F',' '
                NR > 1 && $6 != "NA" {
                    sum[$1] += $6;
                    count[$1]++;
                }
                END {
                    for (et in sum) {
                        printf "| %10s | %7.2f | %13d |\n", et, sum[et]/count[et], count[et];
                    }
                }
            ' "${PPL_CSV_FILE}" | sort -t'|' -k3 -n
            
            echo ""
        fi
        
        echo "## Full Raw Output"
        echo ""
        echo '```'
        cat "${PPL_RESULTS_FILE}.raw"
        echo '```'
        
    } > "${PPL_RESULTS_FILE}"
    
    log_success "PPL results saved to: ${PPL_RESULTS_FILE}"
    log_success "PPL CSV data saved to: ${PPL_CSV_FILE}"
}

################################################################################
# Main Execution
################################################################################

main() {
    section_header "Paper Benchmark Automation - Starting"
    
    log_info "All results will be saved to: ${STATS_DIR}/"
    log_info "Timestamp: ${TIMESTAMP}"
    
    # Execute all steps
    step1_machine_info
    step2_build
    step3_download_convert
    step4_quantize_embeddings
    step5_tune_gemm
    step6_benchmark
    step7_ppl_benchmark
    
    # Final summary
    section_header "All Benchmarks Completed!"
    
    log_success "Results summary:"
    log_success "  - Machine info:     ${MACHINE_INFO_FILE}"
    log_success "  - Benchmark:        ${BENCH_RESULTS_FILE}"
    log_success "  - PPL results:      ${PPL_RESULTS_FILE}"
    log_success "  - PPL CSV:          ${PPL_CSV_FILE}"
    log_success "  - GEMM tuning log:  stats/tuning_log.csv"
    
    echo ""
    log_info "You can find all results in the ${STATS_DIR}/ directory"
}

# Run main function
main "$@"
