#ifndef CUDA_SPECTRAL_SIGNAL_PROCESSOR_SIGNAL_PROCESSOR_H_
#define CUDA_SPECTRAL_SIGNAL_PROCESSOR_SIGNAL_PROCESSOR_H_

#include <array>
#include <string>
#include <vector>

namespace cuda_spectral {

constexpr int kMaxFrameSize = 1024;
constexpr int kNumEqBands = 5;
constexpr int kReductionBlockSize = 256;

struct ProcessorConfig {
  std::string input_dir = "input_data";
  std::string output_dir = "output_data";
  int frame_size = 1024;
  float sample_rate = 16000.0f;
  float noise_threshold = 18.0f;
  std::array<float, kNumEqBands> band_gains = {1.15f, 1.25f, 1.00f, 0.40f, 0.05f};
  int max_files = 0;  // 0 = process all .csv files in input_dir
  bool export_spectrogram = true;
  bool benchmark_cpu = true;
};

struct SignalMetrics {
  std::string filename;
  int num_samples = 0;
  int num_frames = 0;
  float input_rms = 0.0f;
  float output_rms = 0.0f;
  float peak_freq_hz = 0.0f;
  float gpu_kernel_ms = 0.0f;
};

struct BatchSummary {
  int files_processed = 0;
  long long total_samples = 0;
  long long total_fft_frames = 0;
  float total_h2d_ms = 0.0f;
  float total_gpu_compute_ms = 0.0f;
  float total_d2h_ms = 0.0f;
  double cpu_reference_ms = 0.0;
  std::vector<SignalMetrics> per_signal;
};

// Parses CLI arguments into ProcessorConfig. Returns false on --help or error.
bool ParseArguments(int argc, char** argv, ProcessorConfig* config);

// Loads a single-column or comma-separated floating-point signal CSV file.
bool LoadSignalCsv(const std::string& file_path, std::vector<float>* samples);

// Writes a 1D floating-point signal array to a CSV file.
bool SaveSignalCsv(const std::string& file_path,
                   const std::vector<float>& samples);

// Writes a 2D [num_frames x num_bins] log-magnitude spectrogram to CSV.
bool SaveSpectrogramCsv(const std::string& file_path,
                        const std::vector<float>& spectrogram,
                        int num_frames, int num_bins);

// Executes the full cuFFT + custom CUDA kernel pipeline across all input files.
BatchSummary RunBatchPipeline(const ProcessorConfig& config);

}  // namespace cuda_spectral

#endif  // CUDA_SPECTRAL_SIGNAL_PROCESSOR_SIGNAL_PROCESSOR_H_
