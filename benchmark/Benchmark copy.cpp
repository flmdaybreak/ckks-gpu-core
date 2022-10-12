/* Copyright (c) by CryptoLab Inc.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */
#include <iomanip>
#include <time.h>
#include <unistd.h>
#include "public/Test.h"

using namespace ckks;
using namespace std;

class Timer {
 public:
  Timer(const string& name) : name{name} {
    cudaDeviceSynchronize();
    CudaNvtxStart(name);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
  }

  ~Timer() {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    CudaNvtxStop();
    cout << setprecision(3);
    cout << name << ", " << fixed << setprecision(3) << milliseconds << " ms"
         << endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  string name;
  cudaEvent_t start, stop;
};

void ModUpBench() {}

class Benchmark {
 public:
  Benchmark(const Parameter& param) : ckks{param}, param{param} {
    ckks.context.EnableMemoryPool();
    ModUpBench();
    ModDownBench();
    KeyswitchBench();
    PtxtCtxtBatchBench();
  }

  template <typename F, typename R, class... Args>
  void Run(const string& message, R(F::*mf), Args&&... args) {
    for (int i = 0; i < iters; i++) {
      Timer marker(message);
      (ckks.context.*mf)(std::forward<Args>(args)...);
    }
  }

  template <typename Callable, class... Args>
  void Run(const string& message, Callable C, Args&&... args) {
    for (int i = 0; i < iters; i++) {
      Timer marker(message);
      C(std::forward<Args>(args)...);
    }
  }

  void ModUpBench() {
    auto from = ckks.GetRandomPoly();
    ckks.context.is_modup_batched = false;
    Run("ModUp", &Context::ModUp, from);
    ckks.context.is_modup_batched = true;
    Run("FusedModUp", &Context::ModUp, from);
  }

  void ModDownBench() {
    const int num_moduli_after_moddown = param.chain_length_;  // PQ -> Q
    auto from = ckks.GetRandomPolyRNS(param.max_num_moduli_);
    DeviceVector to;
    ckks.context.is_moddown_fused = false;
    Run("ModDown", &Context::ModDown, from, to, num_moduli_after_moddown);
    ckks.context.is_moddown_fused = true;
    Run("FusedModDown", &Context::ModDown, from, to, num_moduli_after_moddown);
  }

  void KeyswitchBench() {
    auto key = ckks.GetRandomKey();
    auto in = ckks.GetRandomPolyAfterModUp(param.dnum_);  // beta = dnum case
    DeviceVector ax, bx;
    ckks.context.is_keyswitch_fused = false;
    Run("KeySwitch", &Context::KeySwitch, in, key, ax, bx);
    ckks.context.is_keyswitch_fused = true;
    Run("FusedKeySwitch", &Context::KeySwitch, in, key, ax, bx);
  }

  void PtxtCtxtBatchBench() {
    int batch_size = 100;
    vector<Ciphertext> op1(batch_size);
    vector<Plaintext>  op2(batch_size);
    vector<Ciphertext> op3(batch_size);
    // setup
    std::cout << "param.chain_length_ = " << std::endl;
    for (int i = 0; i < batch_size; i++) {
      op1[i] = ckks.GetRandomCiphertext();
      op2[i] = ckks.GetRandomPlaintext();
      op3[i] = ckks.GetRandomCiphertext();
    }
    std::cout << "param.chain_length_ = " << param.chain_length_<<std::endl;
    auto MAD = [&](const auto& op1, const auto& op2) {
      Ciphertext accum, out;
      ckks.context.PMult(op1[0], op2[0], accum);
      for (int i = 1; i < batch_size; i++) {
        ckks.context.PMult(op1[i], op2[i], out);
        // ckks.context.Add(accum, out, accum);
      }
    };
    auto MAD1 = [&](const auto& op1, const auto& op3) {
      Ciphertext accum, out;
      for (int i = 0; i < batch_size; i++) {
        ckks.context.Add(op1[i], op3[i], out);
      }
    };
    auto BatchMAD = [&](const auto& op1, const auto& op2) {
      MultPtxtBatch batcher(&ckks.context);
      Ciphertext accum;
      for (int i = 0; i < batch_size; i++) {
        batcher.push(op1[i], op2[i]);
      }
      batcher.flush(accum);
    };
    Run("PtxtCtxtMAD", MAD, op1, op2);
    Run("PtxtCtxtMAD1", MAD1, op1, op3);
    // Run("BatchedPtxtCtxtMAD", BatchMAD, op1, op2);
  }

 private:
  Test ckks;
  Parameter param;
  int iters = 2;
};


class Benchmark1 {
 public:
  Benchmark1(const Parameter& param) : ckks{param}, param{param} {
    ckks.context.EnableMemoryPool();

    PtxtCtxtBatchBench();
  }

  template <typename F, typename R, class... Args>
  void Run(const string& message, R(F::*mf), Args&&... args) {
    for (int i = 0; i < iters; i++) {
      Timer marker(message);
      (ckks.context.*mf)(std::forward<Args>(args)...);
    }
  }

  template <typename Callable, class... Args>
  void Run(const string& message, Callable C, Args&&... args) {
    for (int i = 0; i < iters; i++) {
      Timer marker(message);
      C(std::forward<Args>(args)...);
    }
  }

  void PtxtCtxtBatchBench() {
    int batch_size = 100;
    vector<Ciphertext> op1(batch_size);
    vector<Plaintext>  op2(batch_size);
    vector<Ciphertext> op3(batch_size);
    // setup
    std::cout << "param.chain_length_ = " << std::endl;
    for (int i = 0; i < batch_size; i++) {
      op1[i] = ckks.GetRandomCiphertext();
      op2[i] = ckks.GetRandomPlaintext();
      op3[i] = ckks.GetRandomCiphertext();
    }
    std::cout << "param.chain_length_ = " << param.chain_length_<<std::endl;
    auto MAD = [&](const auto& op1, const auto& op2) {
      Ciphertext accum, out;
      ckks.context.PMult(op1[0], op2[0], accum);
      for (int i = 1; i < batch_size; i++) {
        ckks.context.PMult(op1[i], op2[i], out);
        // ckks.context.Add(accum, out, accum);
      }
    };
    auto MAD1 = [&](const auto& op1, const auto& op3) {
      Ciphertext accum, out;
      for (int i = 0; i < batch_size; i++) {
        ckks.context.Add(op1[i], op3[i], out);
      }
    };
    auto BatchMAD = [&](const auto& op1, const auto& op2) {
      MultPtxtBatch batcher(&ckks.context);
      Ciphertext accum;
      for (int i = 0; i < batch_size; i++) {
        batcher.push(op1[i], op2[i]);
      }
      batcher.flush(accum);
    };
    Run("PtxtCtxtMAD", MAD, op1, op2);
    // Run("PtxtCtxtMAD1", MAD1, op1, op3);
    // Run("BatchedPtxtCtxtMAD", BatchMAD, op1, op2);
  }


 private:
  Test ckks;
  Parameter param;
  int iters = 2;
};


class Benchmark2 {
 public:
  Benchmark2(const Parameter& param) : ckks{param}, param{param} {
    ckks.context.EnableMemoryPool();

    PtxtCtxtBatchBench();
  }

  template <typename F, typename R, class... Args>
  void Run(const string& message, R(F::*mf), Args&&... args) {
    for (int i = 0; i < iters; i++) {
      Timer marker(message);
      (ckks.context.*mf)(std::forward<Args>(args)...);
    }
  }

  template <typename Callable, class... Args>
  void Run(const string& message, Callable C, Args&&... args) {
    for (int i = 0; i < iters; i++) {
      Timer marker(message);
      C(std::forward<Args>(args)...);
    }
  }

  void PtxtCtxtBatchBench() {
    int batch_size = 100;
    vector<Ciphertext> op1(batch_size);
    vector<Ciphertext> op3(batch_size);
    // setup
    std::cout << "param.chain_length_ = " << std::endl;
    for (int i = 0; i < batch_size; i++) {
      op1[i] = ckks.GetRandomCiphertext();
      op3[i] = ckks.GetRandomCiphertext();
    }
    std::cout << "param.chain_length_ = " << param.chain_length_<<std::endl;

    auto MAD1 = [&](const auto& op1, const auto& op3) {
      Ciphertext accum, out;
      for (int i = 0; i < batch_size; i++) {
        ckks.context.Add(op1[i], op3[i], out);
      }
    };
    // Run("PtxtCtxtMAD", MAD, op1, op2);
    Run("PtxtCtxtMAD1", MAD1, op1, op3);
    // Run("BatchedPtxtCtxtMAD", BatchMAD, op1, op2);
  }


 private:
  Test ckks;
  Parameter param;
  int iters = 2;
};

void func1(){
  Benchmark1 bench(PARAM_SMALL_DNUM);
}

void func2(){
  Benchmark2 bench(PARAM_SMALL_DNUM);
}
int main() {
  // Benchmark bench(PARAM_LARGE_DNUM);
  std::cout << "new version" <<std::endl;
  // func1();
  sleep(5);
  std::cout << " func1 is over " <<std::endl;
  func2();

  return 0;
}