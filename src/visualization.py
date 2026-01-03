"""
Visualization utilities for fish detection analysis
"""

import cv2
import os
import matplotlib.pyplot as plt
import re


def natural_sort_key(s):
    """
    Create a key for natural sorting that handles numbers correctly.
    For example: T1_1.txt, T1_2.txt, T1_3.txt instead of T1_1.txt, T1_10.txt, T1_2.txt
    
    Args:
        s (str): String to parse for sorting
    
    Returns:
        list: List of integers and strings for natural sorting
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def get_frame_number(filename):
    """
    Extract frame number from filename handling different formats
    
    Args:
        filename (str): Name of the file to extract frame number from
    
    Returns:
        int: Frame number extracted from filename
    """
    parts = filename.split('_')
    frame_num = int(parts[-1].split('.')[0])
    return frame_num


def plot_detections_over_time(video_path, labels_dir, output_prefix, 
                              tick_interval=10, title="YOLOv9 Detections Over Time"):
    """
    Create a plot of YOLO detections over time from video and label files.
    
    Args:
        video_path (str): Path to the video file
        labels_dir (str): Directory containing YOLO label files
        output_prefix (str): Prefix for output files (e.g., 'T1' will create 'detections_over_timeT1.svg')
        tick_interval (int): Interval in seconds for x-axis ticks (default: 10)
        title (str): Title for the plot
    
    Returns:
        tuple: (times_seconds, labels_count) - Lists of times and detection counts
    """
    # Load video to get frame rate
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    times_seconds = []
    frame_times = []
    labels_count = []
    
    # Process label files
    for label_file in sorted(os.listdir(labels_dir), key=natural_sort_key):
        if label_file.endswith(".txt"):
            try:
                frame_num = get_frame_number(label_file)
                time_in_seconds = frame_num / fps
                
                times_seconds.append(time_in_seconds)
                
                # Format timestamp for labels
                minutes = int(time_in_seconds // 60)
                seconds = int(time_in_seconds % 60)
                timestamp = f"{minutes:02}:{seconds:02}"
                frame_times.append(timestamp)
                
                # Count detections in this frame
                with open(os.path.join(labels_dir, label_file), 'r') as f:
                    lines = f.readlines()
                
                labels_count.append(len(lines))
            except ValueError as e:
                print(f"Error processing file {label_file}: {e}")
                continue
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.scatter(times_seconds, labels_count, label="Detections per Frame", 
                color="b", marker='o')
    
    # Create ticks at specified intervals
    max_time = max(times_seconds)
    tick_positions = list(range(0, int(max_time) + 30, tick_interval))
    tick_labels = [f"{int(t//60):02}:{int(t%60):02}" for t in tick_positions]
    
    plt.xticks(tick_positions, tick_labels, rotation=90)
    plt.xlabel("Time (minutes:seconds)")
    plt.ylabel("Number of Detections")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save outputs
    plt.savefig(f'detections_over_time{output_prefix}.svg', format='svg', bbox_inches='tight')
    plt.savefig(f'detections_over_time{output_prefix}.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
    return times_seconds, labels_count
