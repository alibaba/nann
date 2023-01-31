#pragma once

#include <queue>
#include <mutex>
#include <iostream>

namespace benchmark {

template <typename T>
class Queue {
 public:
  bool Enqueue(T* obj) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (queue_.size() > kMaxQueueSize) {
      return false;
    } else {
      queue_.push(obj);
      return true;
    }
  }

  T* Dequeue() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (queue_.size() == 0) {
      return nullptr;
    } else {
      T* head = queue_.front();
      queue_.pop();
      return head;
    }
  }

  size_t size() {
    std::lock_guard<std::mutex> guard(mutex_);
    return queue_.size();
  }

 protected:
  static const int kMaxQueueSize = 10000;
  std::queue<T*> queue_;
  std::mutex mutex_;
};

}  // namespace benchmark
