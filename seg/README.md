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

데이터 전처리 코드를 실행한다.

```shell
sh ./run_preprocess.sh
```


## Training

For convenience, we provide both [Low resolution config file](configs/mic/flatHR2fishHR_mic_hrda_384.py), [High resolution config file](configs/mic/flatHR2fishHR_mic_hrda_large.py)
of the final MIC(HRDA) on Source(Flat)→Target(Fisheye). A training job can be launched using:

```shell
# Initial Training with Low resolution
python run_experiments.py --config configs/mic/flatHR2fishHR_mic_hrda.py

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
