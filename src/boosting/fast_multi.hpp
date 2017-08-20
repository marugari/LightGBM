#ifndef LIGHTGBM_BOOSTING_FAST_MULTI_H_
#define LIGHTGBM_BOOSTING_FAST_MULTI_H_

#include <LightGBM/boosting.h>
#include "score_updater.hpp"
#include "gbdt.h"

#include <cstdio>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>

namespace LightGBM {
/*!
* \brief DART algorithm implementation. including Training, prediction, bagging.
*/
class Fast_Multi: public GBDT {
public:
  /*!
  * \brief Constructor
  */
  Fast_Multi() : GBDT() { }
  /*!
  * \brief Destructor
  */
  ~Fast_Multi() { }

  void Boosting() override {
    if (objective_function_ == nullptr) {
      Log::Fatal("No object function provided");
    }
    // objective function will calculate gradients and hessians
    int64_t num_score = 0;
    objective_function_->
      GetGradients(GetTrainingScore(&num_score), gradients_.data(), hessians_.data());
    if (class_need_train_orig_.size() == 0) {
      for (auto v : class_need_train_) {
        class_need_train_orig_.push_back(v);
      }
    }
    class_weight_.resize(num_class_);
    class_weight_rank_.resize(num_class_);
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < num_class_; ++k) {
      class_weight_[k] = 0.0;
      for (data_size_t i = 0; i < num_data_; ++i) {
        size_t idx = static_cast<size_t>(num_data_) * k + i;
        class_weight_[k] += std::abs(gradients_[idx]) / num_data_;
      }
    }
    for (int i = 0; i < num_class_; ++i) {
      class_weight_rank_[i] = 0;
      for (int j =0; j < num_class_; ++j) {
        if (class_weight_[i] < class_weight_[j]) {
          class_weight_rank_[i] += 1;
        }
      }
      Log::Info("weight: %f", class_weight_[i]);
      if (class_weight_rank_[i] < gbdt_config_->num_class_train) {
        class_need_train_[i] = class_need_train_orig_[i];
        Log::Info("train class: %d", i);
      } else {
        class_need_train_[i] = false;
      }
    }
  }

private:
  std::vector<score_t> class_weight_;
  std::vector<int> class_weight_rank_;
  std::vector<bool> class_need_train_orig_;
};

}  // namespace LightGBM
#endif   // LightGBM_BOOSTING_FAST_MULTI_H_
