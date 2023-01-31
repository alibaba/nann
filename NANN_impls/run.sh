set -e
set -x

output_root=$1

python main.py --job-type extract_feature --output-root "${output_root}"
python build_hnsw_index.py -i "${output_root}/embeddings/item_embs.npy" -o "${output_root}/index"
python main.py --job-type test --output-root "${output_root}"
python main.py --job-type test_all --output-root "${output_root}"
