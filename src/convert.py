import xml.etree.ElementTree as ET
import json
import urllib.request
import os

XML_PATH = r'.\assets\haarcascade_frontalface_default.xml'
RAW_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'

def ensure_valid_xml():
    need_download = False
    if not os.path.exists(XML_PATH):
        need_download = True
    else:
        with open(XML_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.read(200)
            if '<!DOCTYPE html>' in first_line or 'html' in first_line.lower():
                need_download = True

    if need_download:
        urllib.request.urlretrieve(RAW_URL, XML_PATH)

def convert():
    ensure_valid_xml()

    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    cascade_node = root.find('.//haarcascade_frontalface_default')
    if cascade_node is None:
        cascade_node = root.find('.//cascade')

    size_node = cascade_node.find('size')
    if size_node is not None and size_node.text:
        parts = size_node.text.strip().split()
        width, height = int(parts[0]), int(parts[1])
    else:
        width = int(cascade_node.find('width').text)
        height = int(cascade_node.find('height').text)

    stages = []
    stages_node = cascade_node.find('stages')

    for stage_node in stages_node:
        if not isinstance(stage_node.tag, str):
            continue

        thresh_node = stage_node.find('stage_threshold')
        if thresh_node is None:
            thresh_node = stage_node.find('stageThreshold')

        stage_thresh = float(thresh_node.text) if (thresh_node is not None and thresh_node.text) else 0.0

        classifiers = []
        trees_node = stage_node.find('trees')
        if trees_node is None:
            continue

        for tree_node in trees_node:
            classifier_node = tree_node.find('.//threshold/..')
            if classifier_node is None:
                continue

            thresh = float(classifier_node.find('threshold').text)

            left_node = classifier_node.find('left_val')
            left_val = float(left_node.text) if left_node is not None else 0.0

            right_node = classifier_node.find('right_val')
            right_val = float(right_node.text) if right_node is not None else 0.0

            rects = []
            rects_node = classifier_node.find('feature/rects')
            if rects_node is not None:
                for r in rects_node:
                    if r.text:
                        parts = r.text.strip().split()
                        if len(parts) >= 5:
                            rects.append({
                                "x": int(parts[0]),
                                "y": int(parts[1]),
                                "width": int(parts[2]),
                                "height": int(parts[3]),
                                "weight": float(parts[4])
                            })

            classifiers.append({
                "feature": {"rects": rects},
                "threshold": thresh,
                "left_val": left_val,
                "right_val": right_val
            })

        stages.append({
            "stage_threshold": stage_thresh,
            "classifiers": classifiers
        })

    data = {
        "window_width": width,
        "window_height": height,
        "stages": stages
    }

    with open('face_cascade.json', 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    convert()
