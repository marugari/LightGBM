#ifndef LIGHTGBM_BOOSTING_PART_H_
#define LIGHTGBM_BOOSTING_PART_H_

#include <LightGBM/boosting.h>
#include "score_updater.hpp"
#include "dart.hpp"

#include <cstdio>
#include <vector>
#include <string>
#include <fstream>

namespace LightGBM {
/*!
* \brief PART algorithm implementation. including Training, prediction, bagging.
*/
class PART: public DART {
public:
  /*!
  * \brief Constructor
  */
  PART() : DART() { }
  /*!
  * \brief Destructor
  */
  ~PART() { }

  /*!
  * \brief one training iteration
  */
  bool TrainOneIter(const score_t* gradient, const score_t* hessian) override {
    is_update_score_cur_iter_ = false;
    peeked_score_ = PeekScore();
    bool ret = GBDT::TrainOneIter(gradient, hessian);
    if (ret) {
      return ret;
    }
    bool is_improved = Normalize();
    if (is_improved) {
      if (!gbdt_config_->uniform_drop) {
        tree_weight_.push_back(shrinkage_rate_);
        sum_weight_ += shrinkage_rate_;
      }
    } else {
      int iter_swp = iter_;
      /* Is this bug? */
      iter_ = models_.size();
      Log::Info("Rollback iter: %d", iter_);
      GBDT::RollbackOneIter();
      iter_ = iter_swp - 1;
    }
    return false;
  }

private:
  /*!
  * \brief drop trees based on drop_rate
  */
  void DroppingTrees() {
    drop_index_.clear();
    bool is_skip = random_for_drop_.NextFloat() < gbdt_config_->skip_drop;
    // select dropping tree indices based on drop_rate and tree weights
    if (!is_skip) {
      double drop_rate = gbdt_config_->drop_rate;
      if (!gbdt_config_->uniform_drop) {
        double inv_average_weight = static_cast<double>(tree_weight_.size()) / sum_weight_;
        if (gbdt_config_->max_drop > 0) {
          drop_rate = std::min(drop_rate, gbdt_config_->max_drop * inv_average_weight / sum_weight_);
        }
        for (int i = 0; i < iter_; ++i) {
          if (random_for_drop_.NextFloat() < drop_rate * tree_weight_[i] * inv_average_weight) {
            drop_index_.push_back(num_init_iteration_ + i);
            if (drop_index_.size() >= static_cast<size_t>(gbdt_config_->max_drop)) {
              break;
            }
          }
        }
      } else {
        if (gbdt_config_->max_drop > 0) {
          drop_rate = std::min(drop_rate, gbdt_config_->max_drop / static_cast<double>(iter_));
        }
        for (int i = 0; i < iter_; ++i) {
          if (random_for_drop_.NextFloat() < drop_rate) {
            drop_index_.push_back(num_init_iteration_ + i);
            if (drop_index_.size() >= static_cast<size_t>(gbdt_config_->max_drop)) {
              break;
            }
          }
        }
      }
    }
    // drop trees
    for (auto i : drop_index_) {
      for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
        auto curr_tree = i * num_tree_per_iteration_ + cur_tree_id;
        models_[curr_tree]->Shrinkage(-1.0);
        train_score_updater_->AddScore(models_[curr_tree].get(), cur_tree_id);
      }
    }
    if (drop_index_.empty()) {
      shrinkage_rate_ = gbdt_config_->learning_rate;
    } else {
      shrinkage_rate_ = gbdt_config_->learning_rate / (1.0f + gbdt_config_->learning_rate);
    }
  }

  /*!
  * \brief normalize dropped trees
  * NOTE: num_drop_tree(k), learning_rate(lr), shrinkage_rate_ = lr / (1 + lr)
  *       step 1: shrink tree to -1 -> drop tree
  *       step 2: shrink tree to -lr / (1 + lr) from -1, by lr / (1 + lr)
  *               -> normalize for valid data, 1 - lr / (1 + lr) = 1 / (1 + lr)
  *       step 3: shrink tree to 1 / (1 + lr) from -lr / (1 + lr), by -1 / lr
  *               -> normalize for train data
  *       improved True: tree weight = 1 / (1 + lr) * old_weight
  *       improved False: shrink tree to lr / (1 + lr) from 1 / (1 + lr), by lr
  *               -> normalize for train/valid data, 1 / (1 + lr) + lr / (1 + lr) = 1
  *               -> shrink tree to 1 from lr / (1 + lr), by (1 + lr) / lr
  */
  bool Normalize() {
    double k = static_cast<double>(drop_index_.size());
    for (auto i : drop_index_) {
      for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
        auto curr_tree = i * num_tree_per_iteration_ + cur_tree_id;
        // update validation score
        models_[curr_tree]->Shrinkage(shrinkage_rate_);
        for (auto& score_updater : valid_score_updater_) {
          score_updater->AddScore(models_[curr_tree].get(), cur_tree_id);
        }
        // update training score
        models_[curr_tree]->Shrinkage(-1.0f / gbdt_config_->learning_rate);
        train_score_updater_->AddScore(models_[curr_tree].get(), cur_tree_id);
      }
    }
    double score = PeekScore();
    Log::Info("current: %f\tnew: %f", cur_score_, peek_score_);
    bool is_improved = score > peeked_score_;
    if (is_improved) {
      for (auto i: drop_index_) {
        if (!gbdt_config_->uniform_drop) {
          sum_weight_ -= tree_weight_[i] * shrinkage_rate_;
          tree_weight_[i] *= (1.0f / (1.0f + gbdt_config_->learning_rate));
        }
      }
    } else {
      for (auto i: drop_index_) {
        for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
          auto curr_tree = i * num_tree_per_iteration_ + cur_tree_id;
          models_[curr_tree]->Shrinkage(gbdt_config_->learning_rate);
          for (auto& score_updater : valid_score_updater_) {
            score_updater->AddScore(models_[curr_tree].get(), cur_tree_id);
          }
          train_score_updater_->AddScore(models_[curr_tree].get(), cur_tree_id);
          models_[curr_tree]->Shrinkage(1.0f / shrinkage_rate_);
        }
      }
    }
    return is_improved;
  }

  double PeekScore() {
    double score = 0.0f;
    size_t i = 0;
    for (size_t j = 0; j < valid_metrics_[i].size(); ++j) {
      auto test_scores = valid_metrics_[i][j]->Eval(valid_score_updater_[i]->score(),
                                                    objective_function_);
      score += valid_metrics_[i][j]->factor_to_bigger_better() * test_scores.back();
    }
    return score;
  }

  double peeked_score_;
};

}  // namespace LightGBM
#endif   // LightGBM_BOOSTING_PART_H_
