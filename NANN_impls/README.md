# NANN

NANN离线模块。包括模型训练、item 侧特征抽取、建图、测试检索和复杂模型全量打分的离线指标（precision，recall等）、模型rewrite、和性能压测。

相关链接：

* ATA：https://ata.alibaba-inc.com/articles/220894
* 内部开源代码库：http://gitlab.alibaba-inc.com/alimama-displayads-match-pubilc/NANN
* 学术文章：http://arxiv.org/abs/2202.10226

## Taobao UserBehavior 数据集准备

选择1. 直接下载处理后的tf records，并解压至 ./data。

```bash
wget http://170090.oss-cn-hangzhou-zmf.aliyuncs.com/public/UserBehavior_tfrecords.tar
tar xvf UserBehavior_tfrecords.tar ./
```

选择2. 从 [天池](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649) 下载原始数据集，并运行以下脚本将其转为tfrecords格式 （耗时约十小时）。

```bash
python convert_UB_to_tfrecord.py -i path/to/UserBehavior.csv -o ./data_provider
```

更多选项可通过运行 `python convert_UB_to_tfrecord.py --help` 查看。

## 模型训练

为了加速模型训练，NANN_impls 借助 tf.distribute.MirroredStrategy 实现了单机多卡并行训练。可通过如下命令训练模型：

```bash
eps=0.0
batch_size=800
num_neg=200
output_root=${pwd}/output/adv${eps}_bs${batch_size}_neg${num_neg}
python main.py --job-type train --adv-eps ${eps} --num-neg ${num_neg} --batch-size ${batch_size} --output-root ${output_root}
```

其中`adv-eps` 控制对抗训练过程中扰动向量的大小，为0时不施加对抗扰动；`num-neg`控制采样负样本的数量；更多运行选项可通过运行 `python main.py --help` 查看。
上述命令默认使用本机器所有GPU，可通过`CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py ...`控制NANN_impls可使用哪些GPU训练模型。
训练后，模型文件将保存在`${output_root}/model`目录下。

训练时间取决于 `batch_size` 和 GPU 数量，在 `8 V100, batch size 800, num neg 200` 的条件下，训练时间大约为 11 小时。

PS.
为了方便大家走通流程，我们还提供了[预训练模型](http://270639-public.oss-cn-hangzhou-zmf.aliyuncs.com/public/nann_ub_ckpt_adv3e-5_bs800_neg200.tar)
，下载后解压至`${output_root}/model`目录下，并设置好`${output_root}`系统变量后，即可进行后续操作。

```bash
## 在NANN_impls目录下设置output_root路径
mkdir output
output_root=$(pwd)/output
```

## item 侧特征抽取

抽取item侧特征，将生成 `item_ids.npy, item_embs.npy` 两个文件到 `${output_root}/embeddings` 目录。

```bash
python main.py --job-type extract_feature --output-root ${output_root}
```

## 构建HNSW索引

根据 item embedding (`item_embs.npy`)构建HNSW索引，并将相关结果写到 `${output_root}/embeddings` 目录下。

```bash
export PYTHONPATH=${PYTHONPATH}:$(pwd);
python nann/delivery/build_hnsw_index.py -i ${output_root}/embeddings/item_embs.npy -o ${output_root}/index
```

更多建图选项可通过运行 `python build_hnsw_index --help` 查看。

## 测试检索recall

```bash
python main.py --job-type test --output-root ${output_root}
```

## 测试全量打分recall

```bash
python main.py --job-type test_all --output-root ${output_root}
```

## 模型rewrite

会生成frozen_graph.pb, 0.meta，用于在线部署及inference

```bash
export PYTHONPATH=${PYTHONPATH}:$(pwd);
# 产出 frozen_graph.pb，存储模型结构及参数
python nann/delivery/convert_meta.py --model-dir ${output_root}/model --output-dir ${output_root}/converted_model
# 产出 0.meta，储存检索图
python nann/delivery/build_opt_graph.py --model-dir ${output_root}/model --index-dir ${output_root}/index --item-embs-dir ${output_root}/embeddings --output-dir ${output_root}/converted_model
```

## Inference

```bash
export PYTHONPATH=${PYTHONPATH}:$(pwd);
python nann/delivery/NANN_inference_demo.py --graph-file ${output_root}/converted_model/frozen_graph.pb --meta-path ${output_root}/converted_model/0.meta
```

## benchmark

```bash
../blaze-benchmark/build/benchmark/benchmark benchmark_conf
```

