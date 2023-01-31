#pragma once

#include <memory>
#include <atomic>
#include <thread>

namespace benchmark {

template <typename T>
class DoubleBufferReloader {
 public:
  DoubleBufferReloader() : curr_idx_(0) {
  }

  virtual ~DoubleBufferReloader() {
    obj_buffer_[0].reset();
    obj_buffer_[1].reset();
  }

  std::shared_ptr<T> Instance() {
    return obj_buffer_[curr_idx_.load()];
  }

  /// Update object
  virtual bool Switch() {
    size_t prepare = 1 - curr_idx_.load();
    /// Step1: Wait available
    while (obj_buffer_[prepare].use_count() > 1) {
      std::this_thread::yield();
      continue;
    }

    /// Step2: Write buffer slot
    this->obj_buffer_[prepare].reset(CreateObject());

    /// Step3: Switch slot
    this->curr_idx_ = prepare;

    /// Step4: Release previous buffer slot
    prepare = 1 - curr_idx_.load();
    this->obj_buffer_[prepare].reset();
    return true;
  }

 protected:
  /// Create object by file path
  virtual T* CreateObject() = 0;

 private:
  /// Current index
  std::atomic_size_t curr_idx_;

  /// Double buffer object
  std::shared_ptr<T> obj_buffer_[2];
};

}  // namespace benchmark

