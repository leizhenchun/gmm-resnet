# GMM-ResNet for Synthetic Speech Detection

## step 1：extract LFCC features for all utterances.
run ```asvspoof21\as21_feature.py```

## step 2: train the UBM using matlab.
run ```asvspoof21\matlab\asvspoof21_train_gmm.m```

## step 3: train GMM-ResNet2 and test.
 run ```asvspoof21\as21_gmmresnet2.py```

 or train GMM-ResNet

 run ```asvspoof21\as21_gmmresnet.py```

## Done!

```
@INPROCEEDINGS{10447628,
  author={Lei, Zhenchun and Yan, Hui and Liu, Changhong and Zhou, Yong and Ma, Minglei},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={GMM-ResNet2: Ensemble of Group Resnet Networks for Synthetic Speech Detection}, 
  year={2024},
  volume={},
  number={},
  pages={12101-12105},
  keywords={Voice activity detection;Training;Deep learning;Signal processing;Feature extraction;Acoustics;Speaker recognition;synthetic speech detection;GMM-ResNet2;multi-order GMMs;anti-spoofing},
  doi={10.1109/ICASSP48485.2024.10447628}}
```
```
@inproceedings{lei23_interspeech,
  author={Zhenchun Lei and Yan Wen and Yingen Yang and Changhong Liu and Minglei Ma},
  title={{Group GMM-ResNet for Detection of Synthetic Speech Attacks}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={3187--3191},
  doi={10.21437/Interspeech.2023-1249},
  issn={2958-1796}
}
```
```
@INPROCEEDINGS{9746163,
  author={Lei, Zhenchun and Yan, Hui and Liu, Changhong and Ma, Minglei and Yang, Yingen},
  booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Two-Path GMM-ResNet and GMM-SENet for ASV Spoofing Detection}, 
  year={2022},
  pages={6377-6381},
  keywords={Training;Voice activity detection;Databases;Neural networks;Signal processing;Network architecture;Feature extraction;anti-spoofing;ResNet;SENet;automatic speaker verification},
  doi={10.1109/ICASSP43922.2022.9746163}}
```
```
@inproceedings{wen22_interspeech,
  author={Yan Wen and Zhenchun Lei and Yingen Yang and Changhong Liu and Minglei Ma},
  title={{Multi-Path GMM-MobileNet Based on Attack Algorithms and Codecs for Synthetic Speech and Deepfake Detection}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={4795--4799},
  doi={10.21437/Interspeech.2022-10312},
  issn={2958-1796}
}
```
```
  @inproceedings{lei20_interspeech,
  author={Zhenchun Lei and Yingen Yang and Changhong Liu and Jihua Ye},
  title={{Siamese Convolutional Neural Network Using Gaussian Probability Feature for Spoofing Speech Detection}},
  year=2020,
  booktitle={Proc. Interspeech 2020},
  pages={1116--1120},
  doi={10.21437/Interspeech.2020-2723},
  issn={2958-1796}
}
```
