## Emotional Text-To-Speech Based on Mutual-Information-Guided Emotion-Timbre Disentanglement
### Introduction
Implementation for our recent [paper](https://www.arxiv.org/abs/2510.01722).

**Abstract:** Current emotional Text-To-Speech (TTS) and style transfer methods rely on reference encoders to control global style or emotion vectors, but do not capture nuanced acoustic details of the reference speech. To this end, we propose a novel emotional TTS method that enables fine-grained phoneme-level emotion embedding prediction while disentangling intrinsic attributes of the reference speech. The proposed method employs a style disentanglement method to guide two feature extractors, reducing mutual information between timbre and emotion features, and effectively separating distinct style components from the reference speech. Experimental results demonstrate that our method outperforms baseline TTS systems in generating natural and emotionally rich speech. This work highlights the potential of disentangled and fine-grained representations in advancing the quality and flexibility of emotional TTS systems.

### Data preparation (ESD)
1) Prepare MFA inputs
```bash
python3 prepare_align.py config/ESD/preprocess.yaml
```
2) Run Montreal Forced Aligner (MFA)
```bash
./montreal-forced-aligner/bin/mfa_align raw_data/ESD/ lexicon/librispeech-lexicon.txt english preprocessed_data/ESD
```
3) Preprocess features
```bash
python3 preprocess.py config/ESD/preprocess.yaml
```
Then, change the value of `path['preprocessed_path']` in `config/ESD/preprocess.yaml` to the actual path.


### Training overview
- All artifacts are saved under `./output/<timestamp_version>/` including `ckpt/`, `log/`, `results/`, and snapshot of configs.
- We train in two stages:
  - Stage 1 (FS2 only): Train a clean FastSpeech2 on ESD Neutral subset.
  - Stage 2 (Full Model): Initialize the encoder from Stage 1 FS2, freeze it, and train the full model with the StyleEncoder (Timbre Extractor + Emotion Extractor).

#### Stage 1: Train FS2 on ESD Neutral
```bash
python3 train_stage1.py -p config/ESD/preprocess.yaml -m config/ESD/FS2.yaml -t config/ESD/train.yaml
```
Notes:
- Set batch/steps in `config/ESD/train.yaml` as needed.

#### Stage 2: Train Full Model with StyleEncoder
1) Put the Stage 1 version name and checkpoint step into `config/ESD/train.yaml` fields (the version directory is like `<timestamp>_FS2`):
```yaml
stage1_version: "<stage1_version_name>"
stage1_ckpt_step: 20000  # or your best ckpt step
```
2) Run Stage 2 training:
```bash
python3 train_stage2.py -p config/ESD/preprocess.yaml -m config/ESD/model.yaml -t config/ESD/train.yaml
```


### Implementation References
- [ming024's FastSpeech implementation](https://github.com/ming024/FastSpeech2)
- [KinglittleQ's GST implementation](https://github.com/KinglittleQ/GST-Tacotron)


### Citation
```bibtex
@misc{yang2025emotionaltexttospeechbasedmutualinformationguided,
  title        = {Emotional Text-To-Speech Based on Mutual-Information-Guided Emotion-Timbre Disentanglement},
  author       = {Jianing Yang and Sheng Li and Takahiro Shinozaki and Yuki Saito and Hiroshi Saruwatari},
  year         = {2025},
  eprint       = {2510.01722},
  archivePrefix= {arXiv},
  primaryClass = {cs.SD},
  url          = {https://arxiv.org/abs/2510.01722}
}
```
