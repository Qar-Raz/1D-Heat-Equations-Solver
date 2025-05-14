//  Copyright (c) 2023 AUTHORS
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <cmath> // For fmod if needed, but direct int modulo is better

#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(err)                                                        \
  do {                                                                         \
    cudaError_t err_ = (err);                                                  \
    if (err_ != cudaSuccess) {                                                 \
      std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__             \
                << ": " << cudaGetErrorString(err_) << std::endl;              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

using std::size_t;

// Global constants (original C++ values)
// const size_t ghosts = 1; // Not directly used in CUDA kernel in the same way
size_t nx = 1000000;
const double k_const = 0.4; // Renamed to avoid conflict with kernel parameter
size_t nt = 10000;
const double dt = 1.0;
const double dx = 1.0;
// size_t threads = 4; // Not applicable in the same way for CUDA, refers to CUDA threads per block or total threads

void pr(const std::vector<double>& total) {
  std::cout << "[";
  for(size_t i=0; i<total.size(); i++) {
    std::cout << " " << total[i];
  }
  std::cout << " ]" << std::endl;
}

// CUDA Kernel for the heat equation update step
__global__ void heat_equation_kernel(double* u_new, const double* u_old, size_t N, double k_val, double dt_val, double dx_val) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        // Periodic boundary conditions:
        // For u_old[i-1]: if i is 0, neighbor is N-1
        // For u_old[i+1]: if i is N-1, neighbor is 0
        double u_left   = u_old[(i == 0) ? (N - 1) : (i - 1)];
        double u_center = u_old[i];
        double u_right  = u_old[(i == N - 1) ? 0 : (i + 1)];

        u_new[i] = u_center + k_val * dt_val / (dx_val * dx_val) * (u_left + u_right - 2.0 * u_center);
    }
}


int main(int argc, char **argv) {
  // Values are hardcoded at the top of the file

  std::cout << "Initializing data for CUDA version..." << std::endl;
  std::vector<double> h_u(nx); // Host copy for initialization and final result

  // Initialize data on the host (same as original C++ worker initialization logic)
  // Original worker logic: data.at(n) = n + lo + off; where off=1
  // and lo was relative to worker.
  // Globally, grid point 'i' was initialized to 'i + 1.0'.
  for (size_t i = 0; i < nx; ++i) {
    h_u[i] = static_cast<double>(i + 1.0);
  }

  double* d_u;
  double* d_unew;

  CUDA_CHECK(cudaMalloc(&d_u, nx * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_unew, nx * sizeof(double)));

  CUDA_CHECK(cudaMemcpy(d_u, h_u.data(), nx * sizeof(double), cudaMemcpyHostToDevice));
  // d_unew doesn't need initialization as it's an output buffer

  // CUDA execution parameters
  // The 'threads' variable from C++ is not directly analogous.
  // We choose threadsPerBlock for CUDA kernel launch.
  int threadsPerBlock = 256;
  int numBlocks = (nx + threadsPerBlock - 1) / threadsPerBlock;

  std::cout << "Starting CUDA computation..." << std::endl;
  std::cout << "Grid size (nx): " << nx << std::endl;
  std::cout << "Time steps (nt): " << nt << std::endl;
  std::cout << "Threads per block: " << threadsPerBlock << std::endl;
  std::cout << "Number of blocks: " << numBlocks << std::endl;


  auto t1 = std::chrono::high_resolution_clock::now();

  for (size_t t = 0; t < nt; ++t) {
    heat_equation_kernel<<<numBlocks, threadsPerBlock>>>(d_unew, d_u, nx, k_const, dt, dx);
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

    // Swap pointers for the next iteration
    double* temp = d_u;
    d_u = d_unew;
    d_unew = temp;
  }
  CUDA_CHECK(cudaDeviceSynchronize()); // Wait for all GPU operations to complete

  auto t2 = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 -t1).count()*1e-9;
  std::cout << "elapsed: " << elapsed << std::endl;

  // Copy result back to host
  // After the loop, d_u holds the latest computed values
  CUDA_CHECK(cudaMemcpy(h_u.data(), d_u, nx * sizeof(double), cudaMemcpyDeviceToHost));

  if (nx <= 20) {
    pr(h_u);
  }

  // Performance data logging (similar to original)
  if (!std::filesystem::exists("perfdata.csv")) {
    std::ofstream f("perfdata.csv");
    f << "lang,nx,nt,threads_per_block,dt,dx,total_time,gflops" << std::endl; // Changed "threads" to "threads_per_block"
    f.close();
  }
  std::ofstream f("perfdata.csv", std::ios_base::app);
  // Estimate GFLOPs: 5 ops (2 add, 2 sub, 1 mul) per point per time step in the stencil update `u_center + C * (u_left + u_right - 2.0 * u_center)`
  // k_val * dt_val / (dx_val * dx_val) is a constant factor (C) calculated once per step (or precalculated).
  // Stencil: (u_left + u_right - 2.0 * u_center) -> 1 add, 1 sub, 1 mul (or 2 subs). Let's say 3 ops.
  // Then C * (...) -> 1 mul
  // Then u_center + (...) -> 1 add
  // Total: 5 floating point operations per grid point per time step.
  // The division k_val * dt_val / (dx_val * dx_val) is done once per kernel by each thread, so 3 ops (2 muls, 1 div)
  // total_ops = (5 ops/point/step + 3 ops for const) * nx * nt.
  // Simpler: 5 main stencil ops: (left+right), (sum - 2*center), (factor * diff), (u_old + update_term)
  // Plus the calculation of the factor k*dt/(dx*dx) -> 2 muls, 1 div = 3 ops. This is done by each thread.
  // So, 8 FLOPs per point per time step.
  double gflops = (8.0 * nx * nt) / (elapsed * 1e9);
  f << "cuda," << nx << "," << nt << "," << threadsPerBlock << "," << dt << "," << dx << "," << elapsed << "," << gflops << std::endl;
  f.close();

  std::cout << "CUDA computation finished. Results written to h_u." << std::endl;
  if (nx <= 20) {
      std::cout << "Final state: ";
      pr(h_u);
  }


  // Free GPU memory
  CUDA_CHECK(cudaFree(d_u));
  CUDA_CHECK(cudaFree(d_unew));

  return 0;
}