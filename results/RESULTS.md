Trained 10500 iterations at batch 4 = 14.12 epochs, 512x1024 crops, evaluated at 1024x2048 on all 500 validation images.

### Per-class IoU

| Class | Pixel share % | IoU % |
| --- | --- | --- |
| road | 37.651 | 97.89 |
| sidewalk | 5.405 | 83.44 |
| building | 21.919 | 91.68 |
| wall | 0.733 | 51.73 |
| fence | 0.820 | 57.09 |
| pole | 1.479 | 63.85 |
| traffic light | 0.197 | 72.47 |
| traffic sign | 0.665 | 79.56 |
| vegetation | 17.324 | 92.26 |
| terrain | 0.832 | 62.29 |
| sky | 3.355 | 94.16 |
| person | 1.299 | 81.79 |
| rider | 0.215 | 59.97 |
| car | 6.514 | 93.69 |
| truck | 0.301 | 64.31 |
| bus | 0.389 | 78.85 |
| train | 0.113 | 74.15 |
| motorcycle | 0.080 | 64.95 |
| bicycle | 0.709 | 77.26 |
| **mIoU** | | **75.86** |
| Pixel accuracy | | 95.70 |

### Grouped

| Group | Mean IoU % |
| --- | --- |
| Large regions | 86.95 |
| Mid-size objects | 69.97 |
| Thin and small | 71.41 |

### Efficiency

| Measurement | Value | Conditions |
| --- | --- | --- |
| Parameters, total | 26.78 M | |
| Encoder / decoder / heads | 24.85 M / 1.913 M / 19.5 K | decoder matches the published 1.91M |
| Peak GPU memory, allocated | 4.692 GB | batch 2, 512x1024, AMP |
| Peak GPU memory, reserved | 5.025 GB | same |
| Step time, median | 0.2897 s | compute only, single GPU |
| Step time, logged | 0.546 s | incl. data loading, single GPU |
| Inference latency | 0.4863 s | 1024x2048, batch 1, AMP |
| Inference latency | 0.0546 s | 512x1024, batch 1, AMP |
| GPU | Tesla T4 | torch 2.11.0+cu128, CUDA 12.8 |

### Supervision cost (measured)

| Supervision | Step time s | Peak alloc GB |
| --- | --- | --- |
| deep_supervision +aux_native | 0.2897 | 4.692 |
| deep_supervision | 0.4656 | 5.807 |
| mutation | 0.8536 | 9.184 |
| last_layer | 0.2901 | 4.905 |

### Per-head mIoU

| Head | Stride | mIoU % | Large % | Thin % |
| --- | --- | --- | --- | --- |
| p4 | 32 | 54.20 | 71.85 | 38.40 |
| p3 | 16 | 63.94 | 78.28 | 53.21 |
| p2 | 8 | 70.26 | 81.96 | 63.84 |
| p1 | 4 | 74.51 | 84.35 | 71.44 |

### Where thin-class pixels go

| Class | Absorbed by | Share of its ground truth |
| --- | --- | --- |
| pole | building | 12.0% |
| pole | vegetation | 4.9% |
| traffic light | building | 6.3% |
| traffic light | vegetation | 4.1% |
| traffic sign | building | 6.9% |
| traffic sign | vegetation | 2.2% |
| rider | person | 8.0% |
| rider | bicycle | 5.6% |
| motorcycle | bicycle | 4.2% |
| motorcycle | car | 3.3% |
| bicycle | rider | 3.2% |
| bicycle | building | 1.7% |
| person | rider | 2.8% |
| person | building | 2.6% |