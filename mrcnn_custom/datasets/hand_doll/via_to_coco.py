from fileinput import filename
import os
import sys
import json
import datetime
import numpy as np
import skimage

## coco
cocodataset_dir = "./test_coco"

## via
viadataset_dir = "./test_via"

subset = "test"
assert subset in ["test", "val"]
# vdataset_dir = os.path.join(via_dataset_dir, subset)
print(viadataset_dir)
annotations = json.load(open(os.path.join(viadataset_dir, "via_region_data.json")))
annotations = list(annotations.values())
print(annotations)
print("================")
annotations = list(a for a in annotations if a['regions'])
for a in annotations:
    if type(a['regions']) is dict:
        polygons = [r['shape_attributes'] for r in a['regions'].values()]
    else:
        polygons = [r['shape_attributes'] for r in a['regions']] 
    print(str(polygons)+"\n")
    
    image_path = os.path.join(viadataset_dir, a['filename'])
    # image = skimage.io.imread(image_path)
    # height, width = image.shape[:2]
    print(str(a['filename'])+"\n")
    # print(str(image_path)+"\n")
    # print("height:{}, width:{}".format(height, width))

print("============ below is coco ============")
annotations = json.load(open(os.path.join(cocodataset_dir, "annotations.json")))
# annotations = list(annotations.values())
annot = annotations["annotations"]
filenames = [imgs_info["file_name"] for imgs_info in annotations["images"]]
filenames = [fn.split("/")[1] for fn in filenames]
print(filenames)

# image_id to annot_id
img_to_annot = {}
for a in annot:
    img_id = a["image_id"]
    if img_id in img_to_annot:
        img_to_annot[img_id].append(a["id"])
    else:
        img_to_annot[img_id] = [a["id"]]
print(img_to_annot)
image_ids = list(img_to_annot.keys())
print(image_ids)

for img_id in image_ids:
    new_annot = []
    for a_id in img_to_annot[img_id]:
        points = [int(item) for item in annot[a_id]["segmentation"][0]]
        points_x = points[::2]
        points_y = points[1::2]
        new_dict = {'name': 'polygon', 'all_points_x': points_x, 'all_points_y': points_y}
        new_annot.append(new_dict)
        # print(type(annot[a_id]["segmentation"]))
    print(new_annot)

    print("")
    print(filenames[img_id])

# for a in annot :
#     # print(a)
    
#     points = [int(points) for points in a["segmentation"][0]]
#     points_x = points[::2]
#     points_y = points[1::2]
#     print(points)
#     print(points_x)
#     print(points_y)
#     print("")

# print("")
# print(filenames)

