"""
Data utilities for fish behaviour detection project
"""

import os


def download_roboflow_dataset(api_key, workspace, project_name, version=1, format="yolov9"):
    """
    Download dataset from Roboflow
    
    Args:
        api_key (str): Roboflow API key
        workspace (str): Workspace name
        project_name (str): Project name
        version (int): Dataset version
        format (str): Export format (default: yolov9)
    
    Returns:
        Dataset object from Roboflow
    """
    from roboflow import Roboflow
    
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version_obj = project.version(version)
    dataset = version_obj.download(format)
    
    return dataset


def count_labels_in_directory(labels_dir):
    """
    Count total number of labels across all files in a directory
    
    Args:
        labels_dir (str): Path to directory containing label files
    
    Returns:
        dict: Dictionary with filename as key and label count as value
    """
    label_counts = {}
    
    for label_file in os.listdir(labels_dir):
        if label_file.endswith(".txt"):
            filepath = os.path.join(labels_dir, label_file)
            with open(filepath, 'r') as f:
                lines = f.readlines()
            label_counts[label_file] = len(lines)
    
    return label_counts
