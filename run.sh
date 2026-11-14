#!/usr/bin/env bash
set -e

mkdir -p artifacts

echo "Step 1: Compiling CUDA Spectral Signal Processor..."
make build

echo "Step 2: Generating 256 noisy multi-harmonic CSV signals (4.19 MSamples)..."
python3 scripts/signal_tools.py --mode generate --input_dir input_data --count 256 --samples 16384 --sample_rate 16000

echo "Step 3: Running batched cuFFT + shared/constant memory GPU pipeline..."
./bin/spectral_processor --input_dir input_data --output_dir output_data --frame_size 1024 --noise_threshold 18.0 | tee artifacts/execution.log

echo "Step 4: Collecting sample CSVs, metrics, and PNG visualizations into proof_of_execution.tar.gz..."
cp input_data/signal_0000.csv artifacts/sample_signal_0000_input.csv
cp output_data/signal_0000.csv artifacts/sample_signal_0000_filtered.csv
cp output_data/signal_0000_spectrogram.csv artifacts/sample_signal_0000_spectrogram.csv
cp output_data/batch_metrics.csv artifacts/batch_metrics.csv
python3 scripts/signal_tools.py --mode visualize --input_dir input_data --output_dir output_data --artifact_dir artifacts
tar -czvf proof_of_execution.tar.gz artifacts/

echo "Done! Upload 'proof_of_execution.tar.gz' to Coursera."
