# Analysis Script Example

This script demonstrates how to use the visualization module to analyze YOLO detection results.

## Usage

```python
from src.visualization import plot_detections_over_time

# Example for Video T1
video_path = "/path/to/video/T1.mp4"
labels_dir = "/path/to/labels/T1"

plot_detections_over_time(
    video_path=video_path,
    labels_dir=labels_dir,
    output_prefix="T1",
    tick_interval=10,
    title="Fish Detections Over Time - Video T1"
)
```