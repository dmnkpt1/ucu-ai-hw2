---
title: Homework 2 - Multiple Object Video Detection
author: Dominika Petrus, Andrii Kryvyi
course: Artificial Intelligence · UCU
date: 2026-11-16
---

## Methodology and motivation

### Data preparation 

The data was given in 2 formats - YOLO and COCO. So firstly we took random 50 frames and checked the consistency between these to formats, so as result: we saw that all frames match, and the boxes lie on the objects correctly. Also check that labels are clean, there are no empty boxes, dublicates or boxes out of frames. About format: the torchvision models (Faster R-CNN, RetinaNet) use boxes from COCO in pixels, and Ultralytics (YOLO, RT-DETR) use YOLO format. Legacy notebooks took YOLO files directly, but current notebooks create their own from COCO (difference between these 2 types of notebooks is explained in Validation design).

In general, the objects that we need to detect are very small. So we downscale frames to 1280 (when standart from Ultralytics is 640), because when the size is 640 about 53.6% of objects will be smaller than 16 px, but when we resize to 1280 this percentage drops to 9.7%. Also in anchor-based models (RetinaNet and Faster R-CNN) add anchors of size 16 px (because standart is 32 px) 
([EDA, sections 4.1 and 4.2][eda])

Also the dataset classes are imbalanced:

| Class | Train | Val | Train, % | Val, % |
|---|---:|---:|---:|---:|
| car | 505,301 | 31,821 | 45.7 | 27.9 |
| person | 328,701 | 50,312 | 29.7 | 44.1 |
| motor | 102,819 | 12,025 | 9.3 | 10.5 |
| van | 46,940 | 6,842 | 4.2 | 6.0 |
| tricycle | 41,349 | 5,487 | 3.7 | 4.8 |
| bicycle | 40,255 | 6,022 | 3.6 | 5.3 |
| truck | 30,498 | 1,359 | 2.8 | 1.2 |
| bus | 9,653 | 264 | 0.9 | 0.2 |

*Source: [EDA, section 3.1][eda]*

We can see that car class dominates 45.7% in train over 0.9 % of bus class. Also other classes are in the minority if we compare to car and person classes. In val. part we have only 264 buses. So the mAP will be smaller due to small AP for the minority classes and also mAP is just average metrics for all classes. So to see the classes that the model distinguish badly, we calculate AP for each class separately. 

The other problem is ignored regions (regions that are not labeled) because they teach model to think that real object are just background noise. To handle it  so we painted them on with the mean color of the image during training, so models do not learn them as background. For validation use parameter `iscrowd` = 1, so the detections is these parts do not calculate as correct or incorrect. Non-target classes are handled same as ignored regions.
([EDA, section 7][eda])

Consecutive frames are very similar, if we compare them we can see that the same object on the current frame and the next one has a median IoU 0.9. So we decided to take only every 5th frame from train data (because after 5 frames the median IoU falls to 0.59), and the number of frames dropped from 24201 to 4868, it also make runtime smaller.
([EDA, sections 9 and 11][eda])

As from the beginning the annotations were given in COCO  Конвертація анотацій (need to see in notebooks)
([EDA, sections 1.1–2][eda])



Frame density on train data has median 43.0 and val. 37.0, but max numbers are 180 and 116. As standard COCO use only 100 boxes, and in our case about 5% of data have over 100 objects, so we put the limit of 500 boxes for 1 frame.
([EDA, section 5][eda])

For each video we calculated its median size of the objects, average number of objects on frame mean_objects, resolution. The size varies from 24.5 to 86.8 px. A lot of video contains the major part of only 1 class. The resolution is from 1344x756 to 3840x2160. The average object capacity varies from about 6.1 to over 103.4 for one frame. Also validation part has only 7 videos, so the result will depend on the videos that were placed there. 
([EDA, sections 3.3 and 6][eda])


### Validation design

The split for train, validation and test was given from the start. Split is done by video, not by frames, to avoid data leakage. Train - 56 videos, val. - 7, so we didn't change here anything.
So we have two validation approaches: quick-validation(every 5th frame, total - 571) and full validation (2846 val. frames). First one we use to quick check the metrics after each epoch when the model is running, to see the progress or regress(as overfitting), also to choose the best epoch for the final results.  Full validation is used at the end, and calculate mAP when IoU is between 0.5 to 0.95, also has restrictions that we mentioned previously: limit of 500 objects detections for 1 frame (as on Kaggle), and ignored regions are given as `iscrowd` = 1. 

And the metrics that we calculate are: AP50, AP_small, AP for classes, speed (ms for frame). 

First 5 notebooks are done by legacy protocol. In that protocol we didn't paint ignored regions, ground-truth boxes didn't delete, Ultralytics models chose the best epoch using built in score for all val. frames. 
After that we made new current protocol, it includes all what was described above plus fixation of seeds. In Results the legacy notebooks have distinct marks, because we can't fairly compare them with new protocol notebooks. 

### Models

## Results and analysis

## Final model and run instructions

[eda]: https://github.com/dmnkpt1/ucu-ai-hw2/blob/main/notebooks/eda.ipynb