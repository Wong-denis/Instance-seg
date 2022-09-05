# Intern Project: Implementation of Instance-Segmentation

##  Main Goal
Make proper masks on **electrical devices** and **cables** in order to measure correct temperature using thermal camera **in real-time**

## Progress

- We have finished testing on Mask RCNN, Yolact and YolactEdge. YolactEdge is the fastest algorithm we know so far, and it actually has higher mAP than Mask RCNN in our case.
- However, Anchor-based instance segmentation(e.g.Yolact and Mask RCNN) seems to be bad at masking overlapping objects(e.g. cables). 
- We are going to test more algorithms like BlendMask and SOLOv2.

## Algorithm 

- [Mask RCNN](https://github.com/matterport/Mask_RCNN.git)
    - [Mask RCNN for TF2.7](https://github.com/Kamlesh364/Mask-RCNN-TF2.7.0-keras2.7.0.git)
- [Yolact](https://github.com/dbolya/yolact.git)
- [Yolact_edge](https://github.com/haotian-liu/yolact_edge.git)

## Hand and doll detection
### Mask RCNN
- Size of dataset: 
    - training: 131 images 
    - validation: 10 images
- Performance
    - FPS: >30
    - mAP: 75.09
- Details
    - iterations: 6000
    - epochs: 60
    - batch size: 2
    - backbone: R-50-FPN
    - RTX 3050

<!-- ![](https://i.imgur.com/1szdotr.png) -->
<img src="https://i.imgur.com/1szdotr.png" alt="drawing" width="500"/>
<!-- <img src="https://i.imgur.com/iu4c8Sq.png" alt="drawing" width="500"/> -->
<img src="https://i.imgur.com/xrX0Apt.png" alt="drawing" width="500"/>




### Yolact_edge
- Size of dataset: 
    - training: 131 images 
    - validation: 10 images
- Performance
    - FPS: >30
    - mAP: 75.09
- Details
    - iterations: 6000
    - epochs: 352
    - batch size: 7
    - backbone: R-50-FPN
    - RTX 3050

<!-- ![](https://i.imgur.com/EYaoi3t.jpg) -->
<img src="https://i.imgur.com/EYaoi3t.jpg" alt="drawing" width="500"/>
<!-- <img src="https://i.imgur.com/UQo3Ghb.jpg" alt="drawing" width="500"/> -->
![](https://i.imgur.com/MFgIZOQ.jpg)
<img src="https://i.imgur.com/MFgIZOQ.jpg" alt="drawing" width="500"/>



## Wire detection
### Mask RCNN
- Size of dataset: 
    - training: 115 images 
    - validation: 19 images
- Performance
    - FPS: 2~5
    - mAP:
- Details
    - iterations: 1000
    - epochs: 2
    - batch size: 2
    - backbone: R-50-FPN
    - RTX 3050

<!-- ![](https://i.imgur.com/E5vz0NW.png) -->
<!-- ![](https://i.imgur.com/iaRuL7D.png) -->
<img src="https://i.imgur.com/E5vz0NW.jpg" alt="drawing" width="500"/>
<img src="https://i.imgur.com/iaRuL7D.jpg" alt="drawing" width="500"/>



### Yolact:
- Size of dataset: 
    - training: 114 images 
    - validation: 18 images
- Performance
    - FPS: 20~22
    - mAP: 22.52
- Details
    - iterations: 4000
    - epochs: 307
    - batch size: 8
    - backbone: R-50-FPN
    - RTX 3050

<!-- ![](https://i.imgur.com/c3DukgS.png)
![](https://i.imgur.com/kLZIbNb.png) -->



### Yolact_edge:
- Size of dataset: 
    - training: 114 images 
    - validation: 18 images
- Performance
    - FPS: >30
    - mAP: 24.77
- Details
    - iterations: 4000
    - epochs: 307
    - batch size: 8
    - backbone: R-50-FPN
    - RTX 3050


<!-- ![](https://i.imgur.com/PXwmzVy.png)
![](https://i.imgur.com/8io5pN6.jpg) -->
<img src="https://i.imgur.com/PXwmzVy.jpg" alt="drawing" width="500"/>
<img src="https://i.imgur.com/8io5pN6.jpg" alt="drawing" width="500"/>

## Issues
### Overlap objects

#### hands

<img src="https://i.imgur.com/NktVLkP.jpg" alt="drawing" width="500"/>

#### cables
<!-- ![](https://i.imgur.com/08APRS8.jpg)
![](https://i.imgur.com/ltQbLca.jpg) -->
<img src="https://i.imgur.com/08APRS8.jpg" alt="drawing" width="500"/>
<img src="https://i.imgur.com/ltQbLca.jpg" alt="drawing" width="500"/>

### Possible Cause
- Size of dataset
- Anchor-based segmentation

### Possible Solution
- Use larger dataset ( >1K images)
- Use other segmentation methods 
    - Anchor-free segmentation: e.g. [SOLOv2](https://github.com/WXinlong/SOLO.git) 
    - Top-down meets Bottom-up: e.g. [BlendMask](https://github.com/aim-uofa/AdelaiDet/tree/master/configs/BlendMask)









