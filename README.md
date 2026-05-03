# PointNet

PyTorch implementation of [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593) (Qi et al., CVPR 2017).

## Tasks

### 3D Object Classification - ModelNet40

| Metric | Paper | This impl |
|--------|-------|-----------|
| Overall accuracy | 89.2% | 87.97% |

![Classification example](resources/results/obj_class/1.gif)

### Scene Semantic Segmentation - S3DIS

Evaluated on Area 6.

| Metric | Paper | This impl |
|--------|-------|-----------|
| mIoU | 47.71% | 46.28% |

![Segmentation example](resources/results/scene_seg/0.gif)

## Training

```bash
# Classification
python scripts/train_cls

# Scene segmentation
python scripts/train_scene_seg
```

## Visualization

```bash
# Classification
python scripts/visualize_cls --checkpoint obj_class_best.pt

# Scene segmentation
python scripts/visualize_seg --checkpoint scene_seg_best.pt
```
