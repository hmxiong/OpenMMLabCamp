# OpenMMLabCamp Work 3

## Dron_Seg -- Base Work
- Using Segformer
- Batch Size 16 training 40000iter on UAV aerial photography dataset 
- | aAcc:90.95 | mIoU:71.02 | mAcc:80.19 | 
- Average fps of 1 evaluations: 62.62 FPS
- Log file : segformer.log Config file : segformer.py Evaluation indicators : eval.txt
- Module Link: https://pan.baidu.com/s/11-BrgbLaBlZwNUAb6fsrQg  Coded：lmkp
<div align="center">
<img src="Dron_Seg/1.jpg" width="45%" />
<img src="Dron_Seg/2.jpg" width="45%" />
<br/>
<div align="center">
<img src="Dron_Seg/3.jpg" width="45%" />
<img src="Dron_Seg/4.jpg" width="45%" />
<br/>
<div align="left">

## PascalVOC -- Advanced Work
- Using DeepLabV3+ Backbone R50-D8
- Batch Size 16 training 50000iter on PascalVOC2012 
- | aAcc:91.73 | mIoU:63.24 | mAcc:72.62 | 
- Average fps of 1 evaluations: 14.33 FPS
- Log file : deep.log Config file : deep.py Evaluation indicators : eval.txt
- Module Link: https://pan.baidu.com/s/1RChUr6CpDHj7wVfEpY5ovA   Coded：4gtu
<div align="center">
<img src="PascalVOC/1.jpg" width="45%" height = "45%"/>
<img src="PascalVOC/2.jpg" width="45%" height = "45%"/>
<br/>
<div align="center">
<img src="PascalVOC/3.jpg" width="45%" height = "70%"/>
<img src="PascalVOC/4.jpg" width="45%" height = "30%"/>
<br/>

All projects are based on MMSegmentation