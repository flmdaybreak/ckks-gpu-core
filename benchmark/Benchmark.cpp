/* Copyright (c) by CryptoLab Inc.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */
#include <iomanip>
#include <unistd.h>
#include <string>
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
  Benchmark(const Parameter& param, const string &cal_type) : ckks{param}, param{param},type{cal_type} {
    ckks.context.EnableMemoryPool();
    // ModUpBench();
    // ModDownBench();
    // KeyswitchBench();
    if(type == "pmul"){
      std::cout<<"start cal p_mul"<<std::endl;
      PtxtCtxt_P_MUL_BatchBench();
      std::cout<<"end cal p_mul"<<std::endl;
      sleep(10);
    }
    if(type == "cadd"){
      std::cout<<"start cal c_add"<<std::endl;
      CtxtCtxt_add_BatchBench();
      std::cout<<"end cal c_add"<<std::endl;
      sleep(10);
    }
    if(type == "bmul"){
      std::cout<<"start cal b_mul"<<std::endl;
      PtxtCtxt_P_BatchMUL_BatchBench();
      std::cout<<"end cal b_mul"<<std::endl;
      sleep(10);
    }
    
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

  void PtxtCtxt_P_MUL_BatchBench() {
    int batch_size = 50;
    vector<Ciphertext> op1(batch_size);
    vector<Plaintext> op2(batch_size);
    // setup
    for (int i = 0; i < batch_size; i++) {
      op1[i] = ckks.GetRandomCiphertext();
      op2[i] = ckks.GetRandomPlaintext();
      
    }
    auto MAD = [&](const auto& op1, const auto& op2) {
      Ciphertext  out;
      // ckks.context.PMult(op1[0], op2[0], accum);
      for (int i = 1; i < batch_size; i++) {
        ckks.context.PMult(op1[i], op2[i], out);
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
    Run("PtxtCtxt_MUL_MAD", MAD, op1, op2);
    // Run("BatchedPtxtCtxtMAD", BatchMAD, op1, op2);
  }



  void PtxtCtxt_P_BatchMUL_BatchBench() {
    int batch_size = 50;
    vector<Ciphertext> op1(batch_size);
    vector<Plaintext> op2(batch_size);
    // setup
    for (int i = 0; i < batch_size; i++) {
      op1[i] = ckks.GetRandomCiphertext();
      op2[i] = ckks.GetRandomPlaintext();
      
    }
    auto BatchMAD = [&](const auto& op1, const auto& op2) {
      MultPtxtBatch batcher(&ckks.context);
      Ciphertext accum;
      for (int i = 0; i < batch_size; i++) {
        batcher.push(op1[i], op2[i]);
      }
      batcher.flush(accum);
    };
    Run("BatchedPtxtCtxtMAD", BatchMAD, op1, op2);
    // Run("BatchedPtxtCtxtMAD", BatchMAD, op1, op2);
  }


  void PtxtCtxt_add_BatchBench() {
    int batch_size = 50;
    vector<Ciphertext> op1(batch_size);
    vector<Ciphertext> op2(batch_size);
    // setup
    for (int i = 0; i < batch_size; i++) {
      op1[i] = ckks.GetRandomCiphertext();
      op2[i] = ckks.GetRandomCiphertext();
      
    }
    auto MAD_ADD = [&](const auto& op1, const auto& op2) {
      Ciphertext  out;
      // ckks.context.PMult(op1[0], op2[0], accum);
      for (int i = 1; i < batch_size; i++) {
        ckks.context.Add(op1[i], op2[i], out);
      }
    };
    // auto BatchMAD = [&](const auto& op1, const auto& op2) {
    //   MultPtxtBatch batcher(&ckks.context);
    //   Ciphertext accum;
    //   for (int i = 0; i < batch_size; i++) {
    //     batcher.push(op1[i], op2[i]);
    //   }
    //   batcher.flush(accum);
    // };
    Run("PtxtCtxt_ADD_MAD", MAD_ADD, op1, op2);
    // Run("BatchedPtxtCtxtMAD", BatchMAD, op1, op2);
  }

 private:
  Test ckks;
  Parameter param;
  string type;
  int iters = 10;
};

int main(int argc, char **argv) {
  // Benchmark bench(PARAM_LARGE_DNUM);
  string typess = string(argv[1]);
  Benchmark bench(PARAM_SMALL_DNUM,typess);
  // sleep(5);
  return 0;
}