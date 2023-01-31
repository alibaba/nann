# -*- coding: utf-8 -*-

import argparse
import os

import faiss
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nann.logger import get_logger

logger = get_logger("nann.build_hnsw_index")


def parse_opt():
  parser = argparse.ArgumentParser("tree builder")
  parser.add_argument("--item-embs-file", "-i", type=str, required=True,
                      help="input item embedding file")
  parser.add_argument("--output-dir", "-o", type=str, required=True,
                      help="output directory for save index")
  parser.add_argument("--hnsw-start-level", type=int, default=2,
                      help="level to start search")
  parser.add_argument("--hnsw-num-neighbors", type=int, default=32,
                      help="hnsw number of neighbors")
  return parser.parse_args()


def get_vector(vec):
  return np.array([vec.at(i) for i in range(vec.size())])


def build_and_save_index(embeddings, start_level, num_neighbors, output_dir):
  index = faiss.IndexHNSWFlat(embeddings.shape[1], num_neighbors)
  index.add(embeddings)
  hnsw = index.hnsw
  offsets = get_vector(hnsw.offsets)
  neighbors = get_vector(hnsw.neighbors)
  cum_nneighbor_per_level = get_vector(hnsw.cum_nneighbor_per_level)
  levels = get_vector(hnsw.levels)

  # saving enter points
  enter_points_file = os.path.join(output_dir, "enter_points.npy")
  logger.info(f"dumping enter points to {enter_points_file}")
  enter_points, = np.nonzero(levels > start_level)
  np.save(enter_points_file, enter_points)

  # saving index graph
  for level in range(0, start_level):
    values = []
    row_splits = [0]
    for idx in range(embeddings.shape[0]):
      if level >= levels[idx]:
        neighbors_i = np.array([])
      else:
        offset = offsets[idx]
        start_offset = offset + cum_nneighbor_per_level[level]
        end_offset = offset + cum_nneighbor_per_level[level + 1]
        neighbors_i = neighbors[start_offset: end_offset]
        neighbors_i = neighbors_i[neighbors_i >= 0]
      values.append(neighbors_i)
      row_splits.append(len(neighbors_i) + row_splits[-1])

    values_file = os.path.join(output_dir, f"neighbors_level_{level}_values.npy")
    row_splits_file = os.path.join(output_dir, f"neighbors_level_{level}_row_splits.npy")
    np.save(values_file, np.concatenate(values, axis=0).astype(np.int64))
    np.save(row_splits_file, np.array(row_splits, dtype=np.int64))


def main():
  args = parse_opt()

  logger.info(f"loading data from {args.item_embs_file}")
  item_embs = np.load(args.item_embs_file)

  logger.info(f"building and saving index to {args.output_dir}")
  os.makedirs(args.output_dir, exist_ok=True)
  build_and_save_index(item_embs, args.hnsw_start_level, args.hnsw_num_neighbors, args.output_dir)


if __name__ == '__main__':
  main()
