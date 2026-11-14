# CUDA Spectral Signal Processor (cuFFT + Shared & Constant Memory)

A GPU-accelerated batch digital signal processing (DSP) pipeline implemented in CUDA C++17 following the **Google C++ Style Guide**. It processes hundreds of time-domain signals in a single execution by combining **batched cuFFT (`cufftPlanMany` R2C/C2R)** with **four custom CUDA kernels** for windowing, 5-band parametric equalization, spectral noise gating, spectrogram extraction, and 50% Overlap-Add (OLA) reconstruction.

## Architecture Overview

1. **`FrameAndHannWindowKernel` (`__constant__` memory):** Slices each input signal into 50%-overlapping frames and multiplies each sample by precomputed Hann window coefficients stored in `__constant__` memory.
2. **`cufftExecR2C` (Batched Forward FFT):** Transforms all frames of a signal into the complex frequency domain in a single batched cuFFT call.
3. **`SpectralEqAndDenoiseKernel` (`__constant__` memory):** Applies a 5-band parametric equalizer gain curve (`kConstBandGains`) and attenuates spectral bins below `noise_threshold` while exporting log-magnitude spectrogram values.
4. **`cufftExecC2R` + `OverlapAddReconstructKernel`:** Transforms filtered frequency bins back to time-domain frames and performs race-free 50% Overlap-Add synthesis with `1/N` normalization.
5. **`SumSquaresSharedKernel` (`__shared__` memory):** Computes input and output signal RMS energy via tree-based shared-memory parallel reduction.

## Build & Run

### Automated End-to-End Execution
```bash
chmod +x run.sh scripts/signal_tools.py
./run.sh
