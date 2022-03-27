# AI model for cervical cancer screening and diagnosis 
![Instance Segmentation Sample](title.jpg)
Cervical cancer (CC) is the fourth most common malignant tumor among women worldwide. Constructing a high-accuracy AI model for cervical cancer screening and diagnosis is important for successful prevention of cervical cancer. In this work, we proposed a robust deep convolutional neural cervical cancer screening method using the whole slide images (WSI) of thinprep cytology test (TCT) slides from 211 cervical cancer and 189 normal patients. To partially solve the problem of high cost and low efficiency of image labeling manually, we used an active learning strategy to promote model build efficiency and accuracy. The sensitivity, specificity, and accuracy of the best models were 100%, 98.5%, and 100% for CC patient identification, respectively. The results also demonstrated that the active learning strategy was superior to the traditional supervised learning strategy in decreasing the cost and enhancing the quality of image labeling has the potential to be applied to the pathological diagnosis of various cancer types.
## 1. Evironment Settings
* Install Labelme  (https://github.com/wkentaro/labelme)
* Configure tensorflow 1.14,  [tf_config.ipynb](code/tf_config.ipynb) shows how to setup tensorflow_gpu-1.14 on a PC machine.
* Clone this github repository to local methidn

## 2. Training Data
[500_tct_labeled_images](https://drive.google.com/file/d/1f-9OFKZjKRsQNmOB1nLykmdV6jCZ5-uK/view?usp=sharing) 



## 3.MaskRCNN Models
* [T1_Model](https://drive.google.com/file/d/1f-9OFKZjKRsQNmOB1nLykmdV6jCZ5-uK/view?usp=sharing) 
* [A1_Model](https://drive.google.com/file/d/1fBnTu--B7tLlJyi-GQ8PURwJa6it4xse/view?usp=sharing) 
* [A2_Model](https://drive.google.com/file/d/1f-9OFKZjKRsQNmOB1nLykmdV6jCZ5-uK/view?usp=sharing) 
* [A3_Model](https://drive.google.com/file/d/1f-9OFKZjKRsQNmOB1nLykmdV6jCZ5-uK/view?usp=sharing) 

##  4. Code
### Cell classification
*   [hpv.py](code/cell_classification/hpv.py)
*   [hpv.py](code/cell_classification/hpv.py)
*   [hpv.py](code/cell_classification/hpv.py)
### patient classfication

## 5. Running
### 5.1 Training Model
* Convensional one step Method

* Activate Learning Method
### 5.2 Predict
### 5.3 Patient classfication




## 9. Testing Data
* Trimming FASTQ files can be found ` <root>/test/fastq` 
* The testing data (SRA_ID: SRP055858, SRA files) was downloaded from https://www.ebi.ac.uk/ena/data/view/PRJNA277309
* For reducing testing time, only 50000 spots from each sample are extracted using the command :
`fastq-dump -X 50000 -Z sra > fastq`
* References
    1. Zheng G, Qin Y, Clark WC, et al. Efficient and quantitative high-throughput tRNA sequencing. Nature Methods. 2015 Sep;12(9):835-837. DOI: 10.1038/nmeth.3478.
    2. Clark WC, Evans ME, Dominissini D, Zheng G, Pan T. tRNA base methylation identification and quantification via high-throughput sequencing. RNA (New York, N.Y.). 2016 Nov;22(11):1771-1784. DOI: 10.1261/rna.056531.116.

## License
Copyright (c) 2020 Quanyuan He Ph.D.

Contact: [hqyone@hotmail.com](mailto:hqyone@hotmail.com)

Released under GPLv3. See
[license](LICENSE.txt) for details.

## 11. Disclaimer
This software is supplied 'as is' without any warranty or guarantee of support. The developers are not responsible for its use, misuse, or functionality. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability arising from, out of, or in connection with this software.
