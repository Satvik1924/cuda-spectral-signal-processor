#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "signal_processor.h"

int main(int argc, char** argv) {
  cuda_spectral::ProcessorConfig config;
  if (!cuda_spectral::ParseArguments(argc, argv, &config)) {
    return EXIT_FAILURE;
  }

  cuda_spectral::BatchSummary summary =
      cuda_spectral::RunBatchPipeline(config);
  if (summary.files_processed == 0) {
    return EXIT_FAILURE;
  }

  // Export per-signal CSV summary report
  const std::string metrics_csv = config.output_dir + "/batch_metrics.csv";
  std::ofstream csv(metrics_csv);
  csv << "filename,num_samples,num_frames,input_rms,output_rms,gpu_compute_ms\n";
  for (const auto& m : summary.per_signal) {
    csv << m.filename << "," << m.num_samples << "," << m.num_frames << ","
        << std::fixed << std::setprecision(5) << m.input_rms << ","
        << m.output_rms << "," << m.gpu_kernel_ms << "\n";
  }
  csv.close();

  const double msamples = static_cast<double>(summary.total_samples) / 1e6;
  const double gpu_throughput =
      (summary.total_gpu_compute_ms > 0.0f)
          ? (msamples / (summary.total_gpu_compute_ms / 1000.0))
          : 0.0;
  const double avg_gpu_per_file_ms =
      summary.total_gpu_compute_ms / summary.files_processed;

  std::cout << "\n============================================================\n"
            << " Batch Execution Summary\n"
            << "============================================================\n"
            << "Signals Processed      : " << summary.files_processed << "\n"
            << "Total Samples          : " << summary.total_samples << " ("
            << std::fixed << std::setprecision(2) << msamples << " MSamples)\n"
            << "Total FFT Frames       : " << summary.total_fft_frames << "\n"
            << "Total H2D Time         : " << std::setprecision(3)
            << summary.total_h2d_ms << " ms\n"
            << "Total GPU Compute Time : " << summary.total_gpu_compute_ms
            << " ms (avg " << avg_gpu_per_file_ms << " ms/signal)\n"
            << "Total D2H Time         : " << summary.total_d2h_ms << " ms\n"
            << "GPU Throughput         : " << std::setprecision(2)
            << gpu_throughput << " MSamples/sec\n";

  if (config.benchmark_cpu && summary.cpu_reference_ms > 0.0) {
    const double speedup = summary.cpu_reference_ms / avg_gpu_per_file_ms;
    std::cout << "CPU Single-Signal Ref  : " << std::setprecision(2)
              << summary.cpu_reference_ms << " ms/signal\n"
              << "GPU Speedup vs. CPU    : " << std::setprecision(1) << speedup
              << "x faster\n";
  }
  std::cout << "Metrics CSV Saved      : " << metrics_csv << "\n";

  return EXIT_SUCCESS;
}
