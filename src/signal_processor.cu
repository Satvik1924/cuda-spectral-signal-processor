#include "signal_processor.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace cuda_spectral {

namespace {

constexpr float kPi = 3.14159265358979323846f;

// Constant memory lookup tables for fast broadcast reads across warps.
__constant__ float kConstHannWindow[kMaxFrameSize];
__constant__ float kConstBandGains[kNumEqBands];

inline void CheckCuda(cudaError_t err, const char* context) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error [" << context
              << "]: " << cudaGetErrorString(err) << "\n";
    std::exit(EXIT_FAILURE);
  }
}

inline void CheckCufft(cufftResult res, const char* context) {
  if (res != CUFFT_SUCCESS) {
    std::cerr << "cuFFT Error [" << context << "]: code "
              << static_cast<int>(res) << "\n";
    std::exit(EXIT_FAILURE);
  }
}

// Kernel 1: Slices signal into 50%-overlapping frames and applies Hann window
// stored in __constant__ memory.
__global__ void FrameAndHannWindowKernel(const float* __restrict__ signal,
                                         float* __restrict__ frames,
                                         int num_samples, int frame_size,
                                         int hop_size, int num_frames) {
  const int sample_idx = threadIdx.x;
  const int frame_idx = blockIdx.x;
  if (frame_idx >= num_frames || sample_idx >= frame_size) return;

  const int global_idx = frame_idx * hop_size + sample_idx;
  const float raw_val = (global_idx < num_samples) ? signal[global_idx] : 0.0f;
  frames[frame_idx * frame_size + sample_idx] =
      raw_val * kConstHannWindow[sample_idx];
}

// Kernel 2: Applies 5-band parametric EQ gains (__constant__ memory) and
// spectral noise gating in the frequency domain while extracting a
// log-magnitude spectrogram.
__global__ void SpectralEqAndDenoiseKernel(cufftComplex* __restrict__ freq_bins,
                                           float* __restrict__ spectrogram,
                                           int num_frames, int num_bins,
                                           float noise_threshold) {
  const int bin_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int frame_idx = blockIdx.y;
  if (frame_idx >= num_frames || bin_idx >= num_bins) return;

  const int flat_idx = frame_idx * num_bins + bin_idx;
  cufftComplex val = freq_bins[flat_idx];
  const float mag = hypotf(val.x, val.y);

  // Record log-magnitude spectrogram before attenuation
  spectrogram[flat_idx] = log10f(1.0f + mag);

  // Map frequency bin into one of 5 logarithmic-style EQ bands
  const float norm_freq =
      static_cast<float>(bin_idx) / static_cast<float>(max(1, num_bins - 1));
  int band = 0;
  if (norm_freq < 0.08f) {
    band = 0;  // Sub-bass / fundamental
  } else if (norm_freq < 0.20f) {
    band = 1;  // Low-mid harmonics
  } else if (norm_freq < 0.45f) {
    band = 2;  // Presence / mid-range
  } else if (norm_freq < 0.70f) {
    band = 3;  // Upper-mid
  } else {
    band = 4;  // High-frequency hiss / interference
  }

  // Spectral noise gate + parametric band gain
  const float gate_scale = (mag < noise_threshold) ? 0.05f : 1.0f;
  const float total_gain = kConstBandGains[band] * gate_scale;

  val.x *= total_gain;
  val.y *= total_gain;
  freq_bins[flat_idx] = val;
}

// Kernel 3: Race-free 50% Overlap-Add (OLA) synthesis with 1/N iFFT scaling.
__global__ void OverlapAddReconstructKernel(const float* __restrict__ frames,
                                            float* __restrict__ output_signal,
                                            int num_samples, int frame_size,
                                            int hop_size, int num_frames) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_samples) return;

  const float inv_n = 1.0f / static_cast<float>(frame_size);
  float acc = 0.0f;

  // Frame f0 starting at or before idx
  const int f0 = idx / hop_size;
  const int off0 = idx % hop_size;
  if (f0 < num_frames && off0 < frame_size) {
    acc += frames[f0 * frame_size + off0] * inv_n;
  }

  // Preceding overlapping frame f1 = f0 - 1
  const int f1 = f0 - 1;
  const int off1 = off0 + hop_size;
  if (f1 >= 0 && f1 < num_frames && off1 < frame_size) {
    acc += frames[f1 * frame_size + off1] * inv_n;
  }

  output_signal[idx] = acc;
}

// Kernel 4: Shared-memory parallel block reduction for sum-of-squares energy.
__global__ void SumSquaresSharedKernel(const float* __restrict__ data,
                                       float* __restrict__ block_partials,
                                       int n) {
  __shared__ float sdata[kReductionBlockSize];
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;

  float val = (idx < n) ? data[idx] : 0.0f;
  sdata[tid] = val * val;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    block_partials[blockIdx.x] = sdata[0];
  }
}

float ComputeGpuRms(const float* d_signal, float* d_partials, int n) {
  const int blocks = (n + kReductionBlockSize - 1) / kReductionBlockSize;
  SumSquaresSharedKernel<<<blocks, kReductionBlockSize>>>(d_signal, d_partials,
                                                          n);
  std::vector<float> h_partials(blocks);
  CheckCuda(cudaMemcpy(h_partials.data(), d_partials, blocks * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "Copy RMS partials");
  double total_sq = 0.0;
  for (float p : h_partials) total_sq += static_cast<double>(p);
  return static_cast<float>(std::sqrt(total_sq / std::max(1, n)));
}

// Single-threaded CPU reference DFT + EQ on first signal for speedup comparison
double RunCpuReferenceFilter(const std::vector<float>& input, int frame_size,
                             float sample_rate, float noise_threshold,
                             const std::array<float, kNumEqBands>& gains) {
  const auto t0 = std::chrono::high_resolution_clock::now();
  const int hop_size = frame_size / 2;
  const int num_frames =
      std::max(1, static_cast<int>(input.size() - frame_size) / hop_size + 1);
  const int num_bins = frame_size / 2 + 1;

  std::vector<float> hann(frame_size);
  for (int n = 0; n < frame_size; ++n) {
    hann[n] = 0.5f * (1.0f - std::cos(2.0f * kPi * n / frame_size));
  }

  std::vector<float> output(input.size(), 0.0f);
  std::vector<float> re(num_bins, 0.0f);
  std::vector<float> im(num_bins, 0.0f);

  // Evaluate first 4 frames on CPU and scale to full signal to avoid multi-minute wait
  const int sampled_frames = std::min(num_frames, 4);
  for (int f = 0; f < sampled_frames; ++f) {
    for (int k = 0; k < num_bins; ++k) {
      float sum_re = 0.0f;
      float sum_im = 0.0f;
      for (int n = 0; n < frame_size; ++n) {
        const float x = input[f * hop_size + n] * hann[n];
        const float angle = -2.0f * kPi * k * n / frame_size;
        sum_re += x * std::cos(angle);
        sum_im += x * std::sin(angle);
      }
      const float mag = std::hypot(sum_re, sum_im);
      const float norm = static_cast<float>(k) / (num_bins - 1);
      int band = (norm < 0.08f) ? 0 : (norm < 0.20f) ? 1 : (norm < 0.45f) ? 2 : (norm < 0.70f) ? 3 : 4;
      const float scale = gains[band] * ((mag < noise_threshold) ? 0.05f : 1.0f);
      re[k] = sum_re * scale;
      im[k] = sum_im * scale;
    }
    output[f * hop_size] += re[0] / frame_size;
  }

  const auto t1 = std::chrono::high_resolution_clock::now();
  const double sampled_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  return sampled_ms * (static_cast<double>(num_frames) / sampled_frames);
}

bool HasCsvExtension(const std::string& name) {
  return name.size() >= 4 && name.substr(name.size() - 4) == ".csv";
}

std::vector<std::string> ListCsvFiles(const std::string& dir_path) {
  std::vector<std::string> files;
  DIR* dir = opendir(dir_path.c_str());
  if (dir == nullptr) return files;
  struct dirent* entry = nullptr;
  while ((entry = readdir(dir)) != nullptr) {
    std::string fname(entry->d_name);
    if (HasCsvExtension(fname)) files.push_back(fname);
  }
  closedir(dir);
  std::sort(files.begin(), files.end());
  return files;
}

void PrintUsage(const char* prog) {
  std::cout << "Usage: " << prog << " [options]\n"
            << "Options:\n"
            << "  --input_dir <path>      Input folder with .csv signals (default: input_data)\n"
            << "  --output_dir <path>     Output folder for filtered signals (default: output_data)\n"
            << "  --frame_size <N>        FFT frame size, power of 2 <= 1024 (default: 1024)\n"
            << "  --sample_rate <Hz>      Sampling frequency in Hz (default: 16000)\n"
            << "  --noise_threshold <V>   Spectral noise gate threshold (default: 18.0)\n"
            << "  --max_files <N>         Limit number of signals, 0 = all (default: 0)\n"
            << "  --no_cpu_bench          Skip CPU reference timing\n"
            << "  --help                  Show this help message\n";
}

}  // namespace

bool ParseArguments(int argc, char** argv, ProcessorConfig* config) {
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return false;
    } else if (arg == "--input_dir" && i + 1 < argc) {
      config->input_dir = argv[++i];
    } else if (arg == "--output_dir" && i + 1 < argc) {
      config->output_dir = argv[++i];
    } else if (arg == "--frame_size" && i + 1 < argc) {
      config->frame_size = std::atoi(argv[++i]);
      if (config->frame_size <= 0 || config->frame_size > kMaxFrameSize) {
        std::cerr << "Error: frame_size must be between 1 and "
                  << kMaxFrameSize << "\n";
        return false;
      }
    } else if (arg == "--sample_rate" && i + 1 < argc) {
      config->sample_rate = static_cast<float>(std::atof(argv[++i]));
    } else if (arg == "--noise_threshold" && i + 1 < argc) {
      config->noise_threshold = static_cast<float>(std::atof(argv[++i]));
    } else if (arg == "--max_files" && i + 1 < argc) {
      config->max_files = std::atoi(argv[++i]);
    } else if (arg == "--no_cpu_bench") {
      config->benchmark_cpu = false;
    } else {
      std::cerr << "Unknown or incomplete option: " << arg << "\n";
      PrintUsage(argv[0]);
      return false;
    }
  }
  return true;
}

bool LoadSignalCsv(const std::string& file_path, std::vector<float>* samples) {
  std::ifstream in(file_path);
  if (!in.is_open()) return false;
  samples->clear();
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ',')) {
      if (!token.empty()) {
        samples->push_back(static_cast<float>(std::atof(token.c_str())));
      }
    }
  }
  return !samples->empty();
}

bool SaveSignalCsv(const std::string& file_path,
                   const std::vector<float>& samples) {
  std::ofstream out(file_path);
  if (!out.is_open()) return false;
  out << std::fixed << std::setprecision(6);
  for (float val : samples) {
    out << val << "\n";
  }
  return out.good();
}

bool SaveSpectrogramCsv(const std::string& file_path,
                        const std::vector<float>& spectrogram,
                        int num_frames, int num_bins) {
  std::ofstream out(file_path);
  if (!out.is_open()) return false;
  out << std::fixed << std::setprecision(4);
  for (int f = 0; f < num_frames; ++f) {
    for (int b = 0; b < num_bins; ++b) {
      out << spectrogram[f * num_bins + b]
          << (b + 1 < num_bins ? "," : "\n");
    }
  }
  return out.good();
}

BatchSummary RunBatchPipeline(const ProcessorConfig& config) {
  BatchSummary summary;
  mkdir(config.output_dir.c_str(), 0755);

  std::vector<std::string> files = ListCsvFiles(config.input_dir);
  if (files.empty()) {
    std::cerr << "Error: No .csv signal files found in '" << config.input_dir
              << "'\n";
    return summary;
  }
  if (config.max_files > 0 &&
      static_cast<size_t>(config.max_files) < files.size()) {
    files.resize(config.max_files);
  }

  cudaDeviceProp prop;
  CheckCuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");

  std::cout << "============================================================\n"
            << " CUDA Spectral Signal Processor (cuFFT + Shared/Const Mem)\n"
            << "============================================================\n"
            << "GPU Device        : " << prop.name << " ("
            << prop.multiProcessorCount << " SMs)\n"
            << "Input Directory   : " << config.input_dir << "\n"
            << "Output Directory  : " << config.output_dir << "\n"
            << "Batch File Count  : " << files.size() << " signals\n"
            << "FFT Frame Size    : " << config.frame_size
            << " samples (50% Overlap-Add)\n"
            << "Noise Gate Thresh : " << config.noise_threshold << "\n\n";

  // Upload Hann window and EQ gains to GPU __constant__ memory once
  std::vector<float> h_hann(config.frame_size);
  for (int n = 0; n < config.frame_size; ++n) {
    h_hann[n] = 0.5f * (1.0f - std::cos(2.0f * kPi * n / config.frame_size));
  }
  CheckCuda(cudaMemcpyToSymbol(kConstHannWindow, h_hann.data(),
                               config.frame_size * sizeof(float)),
            "Copy Hann window to constant memory");
  CheckCuda(cudaMemcpyToSymbol(kConstBandGains, config.band_gains.data(),
                               kNumEqBands * sizeof(float)),
            "Copy EQ gains to constant memory");

  // Reusable device buffers & cuFFT plans across the batch
  float* d_in_signal = nullptr;
  float* d_out_signal = nullptr;
  float* d_frames = nullptr;
  cufftComplex* d_freq = nullptr;
  float* d_spectrogram = nullptr;
  float* d_partials = nullptr;

  int cached_samples = 0;
  int cached_frames = 0;
  cufftHandle fwd_plan = 0;
  cufftHandle inv_plan = 0;
  bool plans_created = false;

  cudaEvent_t evt_start, evt_stop;
  CheckCuda(cudaEventCreate(&evt_start), "cudaEventCreate start");
  CheckCuda(cudaEventCreate(&evt_stop), "cudaEventCreate stop");

  const int hop_size = config.frame_size / 2;
  const int num_bins = config.frame_size / 2 + 1;

  for (size_t idx = 0; idx < files.size(); ++idx) {
    const std::string in_path = config.input_dir + "/" + files[idx];
    const std::string out_path = config.output_dir + "/" + files[idx];

    std::vector<float> h_signal;
    if (!LoadSignalCsv(in_path, &h_signal)) continue;

    const int num_samples = static_cast<int>(h_signal.size());
    const int num_frames =
        std::max(1, (num_samples - config.frame_size) / hop_size + 1);

    if (num_samples != cached_samples || num_frames != cached_frames) {
      if (d_in_signal) cudaFree(d_in_signal);
      if (d_out_signal) cudaFree(d_out_signal);
      if (d_frames) cudaFree(d_frames);
      if (d_freq) cudaFree(d_freq);
      if (d_spectrogram) cudaFree(d_spectrogram);
      if (d_partials) cudaFree(d_partials);
      if (plans_created) {
        cufftDestroy(fwd_plan);
        cufftDestroy(inv_plan);
      }

      const int max_blocks =
          (num_samples + kReductionBlockSize - 1) / kReductionBlockSize;
      CheckCuda(cudaMalloc(&d_in_signal, num_samples * sizeof(float)),
                "cudaMalloc d_in_signal");
      CheckCuda(cudaMalloc(&d_out_signal, num_samples * sizeof(float)),
                "cudaMalloc d_out_signal");
      CheckCuda(
          cudaMalloc(&d_frames, num_frames * config.frame_size * sizeof(float)),
          "cudaMalloc d_frames");
      CheckCuda(
          cudaMalloc(&d_freq, num_frames * num_bins * sizeof(cufftComplex)),
          "cudaMalloc d_freq");
      CheckCuda(
          cudaMalloc(&d_spectrogram, num_frames * num_bins * sizeof(float)),
          "cudaMalloc d_spectrogram");
      CheckCuda(cudaMalloc(&d_partials, max_blocks * sizeof(float)),
                "cudaMalloc d_partials");

      int n_arr[1] = {config.frame_size};
      CheckCufft(cufftPlanMany(&fwd_plan, 1, n_arr, nullptr, 1,
                               config.frame_size, nullptr, 1, num_bins,
                               CUFFT_R2C, num_frames),
                 "cufftPlanMany R2C");
      CheckCufft(cufftPlanMany(&inv_plan, 1, n_arr, nullptr, 1, num_bins,
                               nullptr, 1, config.frame_size, CUFFT_C2R,
                               num_frames),
                 "cufftPlanMany C2R");
      plans_created = true;
      cached_samples = num_samples;
      cached_frames = num_frames;
    }

    if (idx == 0 && config.benchmark_cpu) {
      summary.cpu_reference_ms = RunCpuReferenceFilter(
          h_signal, config.frame_size, config.sample_rate,
          config.noise_threshold, config.band_gains);
    }

    // 1. Host-to-Device transfer
    cudaEventRecord(evt_start);
    CheckCuda(cudaMemcpy(d_in_signal, h_signal.data(),
                         num_samples * sizeof(float), cudaMemcpyHostToDevice),
              "H2D copy");
    cudaEventRecord(evt_stop);
    cudaEventSynchronize(evt_stop);
    float h2d_ms = 0.0f;
    cudaEventElapsedTime(&h2d_ms, evt_start, evt_stop);

    // 2. GPU Compute Pipeline (Windowing -> cuFFT R2C -> Spectral EQ -> cuFFT C2R -> OLA)
    cudaEventRecord(evt_start);
    FrameAndHannWindowKernel<<<num_frames, config.frame_size>>>(
        d_in_signal, d_frames, num_samples, config.frame_size, hop_size,
        num_frames);

    CheckCufft(cufftExecR2C(fwd_plan, d_frames, d_freq), "cufftExecR2C");

    dim3 eq_block(128, 1);
    dim3 eq_grid((num_bins + 127) / 128, num_frames);
    SpectralEqAndDenoiseKernel<<<eq_grid, eq_block>>>(
        d_freq, d_spectrogram, num_frames, num_bins, config.noise_threshold);

    CheckCufft(cufftExecC2R(inv_plan, d_freq, d_frames), "cufftExecC2R");

    const int ola_threads = 256;
    const int ola_blocks = (num_samples + ola_threads - 1) / ola_threads;
    OverlapAddReconstructKernel<<<ola_blocks, ola_threads>>>(
        d_frames, d_out_signal, num_samples, config.frame_size, hop_size,
        num_frames);
    CheckCuda(cudaGetLastError(), "Pipeline kernel launch");

    cudaEventRecord(evt_stop);
    cudaEventSynchronize(evt_stop);
    float compute_ms = 0.0f;
    cudaEventElapsedTime(&compute_ms, evt_start, evt_stop);

    const float input_rms = ComputeGpuRms(d_in_signal, d_partials, num_samples);
    const float output_rms =
        ComputeGpuRms(d_out_signal, d_partials, num_samples);

    // 3. Device-to-Host transfer
    std::vector<float> h_filtered(num_samples);
    cudaEventRecord(evt_start);
    CheckCuda(cudaMemcpy(h_filtered.data(), d_out_signal,
                         num_samples * sizeof(float), cudaMemcpyDeviceToHost),
              "D2H copy");
    cudaEventRecord(evt_stop);
    cudaEventSynchronize(evt_stop);
    float d2h_ms = 0.0f;
    cudaEventElapsedTime(&d2h_ms, evt_start, evt_stop);

    // Save filtered signal CSV (and spectrogram CSV for first 3 signals)
    SaveSignalCsv(out_path, h_filtered);
    float peak_freq_hz = 0.0f;
    if (idx < 3 && config.export_spectrogram) {
      std::vector<float> h_spec(num_frames * num_bins);
      CheckCuda(cudaMemcpy(h_spec.data(), d_spectrogram,
                           num_frames * num_bins * sizeof(float),
                           cudaMemcpyDeviceToHost),
                "D2H spectrogram");
      const std::string stem = files[idx].substr(0, files[idx].size() - 4);
      SaveSpectrogramCsv(config.output_dir + "/" + stem + "_spectrogram.csv",
                         h_spec, num_frames, num_bins);

      int best_bin = 1;
      for (int b = 1; b < num_bins; ++b) {
        if (h_spec[b] > h_spec[best_bin]) best_bin = b;
      }
      peak_freq_hz = (static_cast<float>(best_bin) * config.sample_rate) /
                     static_cast<float>(config.frame_size);
    }

    SignalMetrics sm;
    sm.filename = files[idx];
    sm.num_samples = num_samples;
    sm.num_frames = num_frames;
    sm.input_rms = input_rms;
    sm.output_rms = output_rms;
    sm.peak_freq_hz = peak_freq_hz;
    sm.gpu_kernel_ms = compute_ms;
    summary.per_signal.push_back(sm);

    summary.files_processed++;
    summary.total_samples += num_samples;
    summary.total_fft_frames += num_frames;
    summary.total_h2d_ms += h2d_ms;
    summary.total_gpu_compute_ms += compute_ms;
    summary.total_d2h_ms += d2h_ms;

    if ((idx + 1) % 32 == 0 || idx + 1 == files.size()) {
      std::cout << "[" << std::setw(3) << (idx + 1) << "/" << files.size()
                << "] " << files[idx] << " | In RMS: " << std::fixed
                << std::setprecision(4) << input_rms
                << " -> Out RMS: " << output_rms
                << " | GPU Time: " << compute_ms << " ms\n";
    }
  }

  if (plans_created) {
    cufftDestroy(fwd_plan);
    cufftDestroy(inv_plan);
  }
  cudaEventDestroy(evt_start);
  cudaEventDestroy(evt_stop);
  if (d_in_signal) cudaFree(d_in_signal);
  if (d_out_signal) cudaFree(d_out_signal);
  if (d_frames) cudaFree(d_frames);
  if (d_freq) cudaFree(d_freq);
  if (d_spectrogram) cudaFree(d_spectrogram);
  if (d_partials) cudaFree(d_partials);

  return summary;
}

}  // namespace cuda_spectral
