#pragma once

//find topk in index range [head, tail), output to val_v & idx_v
// safe if len < k
template <typename T>
void safe_topk_in_range(const T* v, int k, int head, int tail, 
                        T* val_v, int* idx_v) {
  int len = tail - head;
  if (len <= k) {
    for (int i = 0; i < len; ++i) {
      idx_v[i] = head + i;
      val_v[i] = v[idx_v[i]];
    }
    return;
  }
  // len > k
  if (k == 1) {
    int idx = head;
    for (int i = head+1; i < tail; ++i) {
      if (v[i] > v[idx]) idx = i;
    }
    idx_v[0] = idx; val_v[0] = v[idx];
  } else if (k == 2) {
    int idx0 = head;
    int idx1 = head+1;
    if (v[idx0] < v[idx1]) std::swap(idx0, idx1);
    for (int i = head+2; i < tail; ++i) {
      if (v[i] > v[idx1]) {
        idx1 = i;
        if (v[idx0] < v[idx1]) std::swap(idx0, idx1);
      }
    }
    idx_v[0] = idx0; val_v[0] = v[idx0];
    idx_v[1] = idx1; val_v[1] = v[idx1];
  } else {
    //TODO: fast-topk algorithm
  } 
}
