# MIC for Camera-Invariant Domain Adaptation

코드가 실행이 되지 않으면 `kyubyung.chae@snu.ac.kr`로 언제든지 연락바랍니다.


## Environment Setup
First, please install cuda version 11.0.3 available at [https://developer.nvidia.com/cuda-11-0-3-download-archive](https://developer.nvidia.com/cuda-11-0-3-download-archive). It is required to build mmcv-full later.

For this project, we used python 3.8.5. We recommend setting up a new virtual
environment:

```shell
python -m venv ~/venv/mic-seg
source ~/venv/mic-seg/bin/activate
```

In that environment, the requirements can be installed with:

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # requires the other packages to be installed first
```

Further, please download the MiT weights from SegFormer using the
following script. If problems occur with the automatic download, please follow
the instructions for a manual download within the script.

```shell
sh tools/download_checkpoints.sh
```


## Checkpoints
MIC에서 훈련된 모델로 초기화하고 훈련을 진행하였음. 링크는 아래와 같음.
(메일로 같이 동봉하였으니, 여기서는 다운로드를 진행하지 않아도 됨.)

Below, we provide checkpoints of MIC(HRDA) for the different benchmarks.
As the results in the paper are provided as the mean over three random
seeds, we provide the checkpoint with the median validation performance here.

* [MIC(HRDA) for GTA→Cityscapes](https://drive.google.com/file/d/1p_Ytxmj8EckYsq6SdZNZJNC3sgxVRn2d/view?usp=sharing)

The checkpoints come with the training logs. Please note that:

* The logs provide the mIoU for 19 classes. For Synthia→Cityscapes, it is
  necessary to convert the mIoU to the 16 valid classes. Please, read the
  section above for converting the mIoU.
* The logs provide the mIoU on the validation set. For Cityscapes→ACDC and
  Cityscapes→DarkZurich the results reported in the paper are calculated on the
  test split. For DarkZurich, the performance significantly differs between
  validation and test split. Please, read the section above on how to obtain
  the test mIoU.




## Dataset Setup

The final folder structure should look like this:

```none
base
├── ...
├── data
│   ├── train_source_image
│   ├── train_source_gt
│   ├── train_target_image
│   ├── val_source_image
│   ├── val_source_gt
│   ├── test_image
│   ├── sample_submission.csv
│   ├── test.csv
│   ├── sample_class_stats_dict.json
│   ├── sample_class_stats.json
│   ├── samples_with_class.json
├── pretrained
│   ├── gtaHR2csHR_mic_hrda_650a8 (다운로드 필요)
│   ├── 231002_0409_flatHR2fishHR_mic_hrda_s2_40209
│   ├── mit_b5.pth
├── ...
```

**Data Preprocessing :** 

**생략; 제공된 json 파일 이용**
Finally, please run the following scripts to convert the label IDs to the
train IDs and to generate the class index for RCS:

```shell
python tools/convert_datasets/flat.py data --gt-dir train_source_gt --nproc 8
```

전체 데이터 전처리 코드는 다음과 같이 실행할 수 있다.

```shell
sh ./run_preprocess.sh
```

그러나 각각 프로세스가 시간이 오래 걸리고 어느 정도 용량을 요구하므로 주의해야 한다.
'생략 가능' 표시가 되어 있으면, 생략하고 진행하여도 된다.

```shell
# 1. Make directory & Copy all Images
train_dir="data/train_source_image/*"
val_dir="data/val_source_image/*"
out_dir="data/train_source_image_all"

mkdir -p $out_dir

cp $train_dir $out_dir
cp $val_dir $out_dir

# 2. Preprocess Ground Truth
python preprocess_gt.py

# 3. Make masks (생략 가능)
python make_mask.py

# 4. Make blended images
python run_blend.py
```


## Training

반드시 [default_runtime_large.py](configs/_base_/default_runtime_large.py)를 수정바랍니다.

For convenience, we provide both [Low resolution config file](configs/mic/flatHR2fishHR_mic_hrda_384.py), [High resolution config file](configs/mic/flatHR2fishHR_mic_hrda_large.py)
of the final MIC(HRDA) on Source(Flat)→Target(Fisheye). A training job can be launched using:

```shell
# Initial Training with Low resolution
python run_experiments.py --config configs/mic/flatHR2fishHR_mic_hrda_384.py

# After Training with High resolution
python run_experiments.py --config configs/mic/flatHR2fishHR_mic_hrda_large.py
```

The logs and checkpoints are stored in `work_dirs/`.


## Evaluation

**반드시 `run_submit.sh`에서 `exp_name`, `config_path`, `checkpoint_path`를 수정해야 함.**

A trained model can be evaluated using:

```shell
sh ./run_submit.sh 
```

혹시나 제출된 pretrained 파일이 아니라 직접 훈련한 모델을 사용하고 싶으면, 해당 경로로 수정바랍니다!

```shell
config_path="${base_path}/work_dirs/local-${type_name}/${exp_name}/${exp_name}.py"
checkpoint_path="${base_path}/work_dirs/local-${type_name}/${exp_name}/${iters}.pth"
```


## Framework Structure

This project is based on [mmsegmentation version 0.16.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0).
For more information about the framework structure and the config system,
please refer to the [mmsegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/index.html)
and the [mmcv documentation](https://mmcv.readthedocs.ihttps://arxiv.org/abs/2007.08702o/en/v1.3.7/index.html).

The most relevant files for MIC are:

* [configs/mic/gtaHR2csHR_mic_hrda.py](configs/mic/gtaHR2csHR_mic_hrda.py):
  Annotated config file for MIC(HRDA) on GTA→Cityscapes.
* [experiments.py](experiments.py):
  Definition of the experiment configurations in the paper.
* [mmseg/models/uda/masking_consistency_module.py](mmseg/models/uda/masking_consistency_module.py):
  Implementation of MIC.
* [mmseg/models/utils/masking_transforms.py](mmseg/models/utils/masking_transforms.py):
  Implementation of the image patch masking.
* [mmseg/models/uda/dacs.py](mmseg/models/uda/dacs.py):
  Implementation of the DAFormer/HRDA self-training with integrated MaskingConsistencyModule

## Acknowledgements

MIC is based on the following open-source projects. We thank their
authors for making the source code publicly available.

* [HRDA](https://github.com/lhoyer/HRDA)
* [DAFormer](https://github.com/lhoyer/DAFormer)
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
* [DACS](https://github.com/vikolss/DACS)
