# Densecap-tensorflow

Implementation of CVPR2017 paper: [A Hierarchical Approach for Generating Descriptive Image Paragraphs](http://cs.stanford.edu/people/ranjaykrishna/im2p/index.html) by ** Jonathan Krause, Justin Johnson, Ranjay Krishna, Fei-Fei Li**

NOTE: This repo is based on [densecap-tensorflow](https://github.com/InnerPeace-Wu/densecap-tensorflow), and it's still buggy.

## Note

**Update 2018.1.27**  

* Following procedures will be adapted for **IM2P** soon.


## Dependencies

To install required python modules by:

```commandline
pip install -r lib/requirements.txt
```

## Preparing data

### Download

[Website of Visual Genome Dataset](http://visualgenome.org/api/v0/api_home.html)

* Make a new directory `VG` wherever you like.
* Download `images` Part1 and Part2, extract `all (two parts)` to directory `VG/images`
* Download `image meta data`, extract to directory `VG/1.2` or `VG/1.0` according to the version you download.
* Download `region descriptions`, extract to directory `VG/1.2` or `VG/1.0` accordingly.
* For the following process, we will refer directory `VG` as `raw_data_path`

### Unlimit RAM

If one has RAM more than 16G, then you can preprocessing dataset with following command.
```shell
$ cd $ROOT/lib
$ python preprocess.py --version [version] --path [raw_data_path] \
        --output_dir [dir] --max_words [max_len]
```

### Limit RAM (Less than 16G)

If one has RAM `less than 16G`.
* Firstly, setting up the data path in `info/read_regions.py` accordingly, and run the script with python. Then it will dump `regions` in `REGION_JSON` directory. It will take time to process more than 100k images, so be patient.
```shell
$ cd $ROOT/info
$ python read_regions --version [version] --vg_path [raw_data_path]
```
* In `lib/preprocess.py`, set up data path accordingly. After running the file, it will dump `gt_regions` of every image respectively to `OUTPUT_DIR` as `directory`.
```shell
$ cd $ROOT/lib
$ python preprocess.py --version [version] --path [raw_data_path] \
        --output_dir [dir] --max_words [max_len] --limit_ram
```

## Compile local libs

```shell
$ cd root/lib
$ make
```

## Train

Add or modify configurations in `root/scripts/dense_cap_config.yml`, refer to 'lib/config.py' for more configuration details.
```shell
$ cd $ROOT
$ bash scripts/dense_cap_train.sh [dataset] [net] [ckpt_to_init] [data_dir] [step]
```

Parameters:
* dataset: `visual_genome_1.2` or `visual_genome_1.0`.
* net: res50, res101
* ckpt_to_init: pretrained model to be initialized with. Refer to [tf_faster_rcnn](https://github.com/endernewton/tf-faster-rcnn) for more init weight details.
* data_dir: the data directory where you save the outputs after `prepare data`.
* step: for continue training. 
    - step 1: fix convnet weights
    - stpe 2: finetune convnets weights
    - step 3: add context fusion, but fix convnets weights
    - step 4: finetune the whole model.

## Demo

Create a directory `data/demo`
```sh
$ mkdir $ROOT/data/demo
```
Then put the images to be tested in the directory and run
```sh
$ cd $ROOT
$ bash scripts/dense_cap_demo.sh [ckpt_path] [vocab_path]
```
It will create html files in `$ROOT/demo`, just click it.
Or you can use the web-based visualizer created by [karpathy](https://github.com/karpathy) by
```sh
$ cd $ROOT/vis
$ python -m SimpleHTTPServer 8181
```
Then point your web brower to [http://localhost:8181/view_results.html](http://localhost:8181/view_results.html).

## TODO:

- [ ] Debugging.


## References

* The Faster-RCNN framework inherited from repo [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn) by [endernewton](https://github.com/endernewton)
* The official repo of [densecap](https://github.com/linjieyangsc/densecap)
* [Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling](https://arxiv.org/abs/1611.01462)
* Official tensorflow models - "im2text".
* Adapted web-based visualizer from [jcjohnson](https://github.com/jcjohnson)'s [densecap repo](https://github.com/jcjohnson/densecap)
