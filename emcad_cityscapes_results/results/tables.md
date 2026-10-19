### Per-class IoU, Cityscapes val (500 images, 1024x2048)

| Class | Pixel share % | IoU % |
| --- | --- | --- |
| road | 37.651 | 77.34 |
| sidewalk | 5.405 | 47.86 |
| building | 21.919 | 78.58 |
| wall | 0.733 | 0.00 |
| fence | 0.820 | 0.00 |
| pole | 1.479 | 9.70 |
| traffic light | 0.197 | 0.00 |
| traffic sign | 0.665 | 27.62 |
| vegetation | 17.324 | 84.73 |
| terrain | 0.832 | 0.00 |
| sky | 3.355 | 73.55 |
| person | 1.299 | 37.67 |
| rider | 0.215 | 0.00 |
| car | 6.514 | 52.49 |
| truck | 0.301 | 0.00 |
| bus | 0.389 | 0.00 |
| train | 0.113 | 0.00 |
| motorcycle | 0.080 | 0.00 |
| bicycle | 0.709 | 1.32 |
| **mIoU** | | **25.83** |
| Pixel accuracy | | 82.84 |

### Grouped

| Group | Mean IoU % |
| --- | --- |
| Large regions | 60.34 |
| Mid-size objects | 8.75 |
| Thin and small | 10.90 |

### Efficiency

| Measurement | Value | Conditions |
| --- | --- | --- |
| Parameters, total | 26.78 M | |
| Encoder / decoder / heads | 24.85 / 1.913 / 19.5 K M | |
| Peak memory, allocated | 4.693 GB | batch 2, 512x1024, AMP |
| Peak memory, reserved | 5.025 GB | same |
| Step time, median | 0.2842 s | compute only, single GPU |
| Step time, logged | 11.593 s | incl. data loading, 4 on 2x T4 |
| Inference latency | 0.5266 s | 1024x2048, batch 1, AMP |
| Inference latency | 0.0565 s | 512x1024, batch 1, AMP |
| Iterations | 100 | batch 4, 0.1 epochs |
| GPU | Tesla T4 | torch 2.11.0+cu128, CUDA 12.8 |