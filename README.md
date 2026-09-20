# Early Bird 🦅

**EarlyBird: Early-Fusion for Multi-View Tracking in the Bird's Eye View**

Torben Teepe, Philipp Wolters, Johannes Gilg, Fabian Herzog, Gerhard Rigoll

[![arxiv](https://img.shields.io/badge/arXiv-2310.13350-red)](https://arxiv.org/abs/2310.13350)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/earlybird-early-fusion-for-multi-view/multi-object-tracking-on-wildtrack)](https://paperswithcode.com/sota/multi-object-tracking-on-wildtrack?p=earlybird-early-fusion-for-multi-view)

## Usage

### Getting Started
1. Install [PyTorch](https://pytorch.org/get-started/locally/) with CUDA support
    ```shell
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
2. Install [mmcv](https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-with-pip) with CUDA support
   ```shell
   pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
   ```
3. Install remaining dependencies
   ```shell
   pip install -r requirements.txt
   ```

#### Training
```shell
python main.py fit -c configs/t_fit.yml \
    -c configs/d_{multiviewx,wildtrack}.yml
```

#### Testing
```shell
python main.py test -c model_weights/config.yaml \
    --ckpt model_weights/model-epoch=35-val_loss=6.50.ckpt
```

## Running on Kaggle / recent environments (this fork)

This fork ([Nhutan410/EarlyBird](https://github.com/Nhutan410/EarlyBird)) keeps the model,
losses, training scheme and evaluation of the original code unchanged; it only makes the
code run out-of-the-box on Kaggle / Colab with current torch, lightning and numpy 2:

- `requirements-kaggle.txt` -- unpinned dependency list (see the comments inside for why
  `lap` -> `lapx`, why `mmcv` is not needed, and why `nuscenes-devkit` is installed with
  `--no-deps`).
- `configs/d_wildtrack.yml` / `configs/d_multiviewx.yml` -- `data_dir` now points to
  `data/Wildtrack` / `data/MultiviewX` (relative to `EarlyBird/`) instead of the author's
  absolute path. Override on the command line with `--data.init_args.data_dir=...`.
- `models/mvdet.py` -- no longer pins the normalisation constants / voxel transform to
  `cuda` (they follow the input tensor's device), so the code also runs on CPU for smoke
  tests. Numerically identical on GPU.
- `data.init_args.drop_ratio` (default 0, no behaviour change) -- partial-annotation training:
  with `drop_ratio: N` the *train* split reads
  `data_dir/drop_annotations/drop_N/annotations_positions/` (same frame files with some
  pedestrians removed); val/test always use the full `annotations_positions/`.
- `utils/console_log.EpochConsoleLogger` -- optional callback printing one line per epoch
  (`--trainer.callbacks+=utils.console_log.EpochConsoleLogger`); the default progress bar is
  silent when stdout is not a terminal.
- `main.py` -- `test` additionally writes `test_metrics.json` next to `moda_pred.txt` etc.
  in the run's log dir.

```shell
pip install -r requirements-kaggle.txt
pip install --no-deps nuscenes-devkit
cd EarlyBird
# data/Wildtrack must be a WRITABLE directory (gt.txt is written into it): on Kaggle create a real
# directory and symlink each entry of the read-only /kaggle/input/... dataset into it.
python main.py fit  -c configs/t_fit.yml -c configs/d_wildtrack.yml --trainer.default_root_dir=/kaggle/working/earlybird_run
python main.py test -c /kaggle/working/earlybird_run/lightning_logs/version_0/config.yaml \
    --ckpt_path /kaggle/working/earlybird_run/lightning_logs/version_0/checkpoints/last.ckpt
```
Resume an interrupted `fit` by adding `--ckpt_path <run>/lightning_logs/version_X/checkpoints/last.ckpt`.
The full Kaggle notebook lives in the capstone project repo
(`notebooks/earlybird/earlybird_wildtrack_tteepe_kaggle.ipynb`).

## Acknowledgement
- [Simple-BEV](https://simple-bev.github.io): Adam W. Harley
- [MVDeTr](https://github.com/hou-yz/MVDeTr): Yunzhong Hou

## Cite
If you use EarlyBird, please use the following BibTeX entry.

```
@InProceedings{teepe2023earlybird,
    author    = {Teepe, Torben and Wolters, Philipp and Gilg, Johannes and Herzog, Fabian and Rigoll, Gerhard},
    title     = {Early{B}ird: Early-Fusion for Multi-View Tracking in the Bird's Eye View},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
    month     = {January},
    year      = {2024},
    pages     = {102-111}
}
```
