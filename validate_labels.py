import os
import glob

def is_yolov8_label_line(line):
    parts = line.strip().split()
    if len(parts) < 6:  # YOLOv8 segmentation: at least class + 4 box + 2 polygon points
        return False
    try:
        class_id = int(parts[0])
        floats = [float(x) for x in parts[1:]]
        # There should be 4 box coords, then an even number of polygon coords
        if len(floats) < 6 or (len(floats) - 4) % 2 != 0:
            return False
        return True
    except ValueError:
        return False

def check_labels_dir(labels_dir):
    errors = []
    for label_file in glob.glob(os.path.join(labels_dir, '**', '*.txt'), recursive=True):
        with open(label_file, 'r') as f:
            for i, line in enumerate(f, 1):
                if not is_yolov8_label_line(line):
                    errors.append(f"{label_file}, line {i}: {line.strip()}")
    return errors

if __name__ == "__main__":
    base_labels_dirs = [
        "data/Crack Segmentation.v1i.yolov8/train/labels",
        "data/Crack Segmentation.v1i.yolov8/valid/labels",
        "data/Crack Segmentation.v1i.yolov8/test/labels",
        "data/crack-seg/labels/train",
        "data/crack-seg/labels/val",
        "data/crack-seg/labels/test"
    ]
    all_errors = []
    for d in base_labels_dirs:
        if os.path.exists(d):
            all_errors.extend(check_labels_dir(d))
    if all_errors:
        print("Label format errors found:")
        for err in all_errors:
            print(err)
    else:
        print("All label files are in YOLOv8 format.")
