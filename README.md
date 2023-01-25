# Open World Papers

Current machine learning algorithms are bound to a closed-world assumption. This means that these models assume the number of categories the model will encounter in the test time is predefined. However, this assumption needs to be revised in the real world. There are many challenges associated with the real-world setting that traditional machine learning models can not resolve. The uncertainty a model can encounter during test time and the strategy to address it have been considered recently under different computer vision branches. 
In tackling the real world, we want to consider layers of uncertainty. In the following sections, we defined each challenge and what field of data will create. We created this repo to list current methods used in recent years to address uncertainty (specifically novel class and open-set) in the world in contemporary top venues like **CVPR, CVPRw, NeurIPS, ICCV, ECCV, ICLR, ICML, BMVC, WACV, TPAMI, AAAI,** and relevant papers from **Arxiv** and other venues. 

Finally, since our primary focus is fine-grained or long-tailed novel action discovery, we also address related works in this area. 
Without further due, let us dive into the fantastic and challenging world of uncertainty, unseens, and unknowns. 

## Contents
- [Introduction](#Introduction)
  - [Unseen Environments](#Unseen-Environments)
  - [Unseen Categories](#Unseen-Categories)
  - [Unknown Categories](#Unknown-Categories)
- [Zero-Shot Learning](#Zero-Shot-Learning)  
- [Out-of-Distribution Detection](#Out-of-Distribution-Detection) 
- [Novel Class Discovery](#Novel-Class-Discovery)
- [Open Vocabulary](#Open-Vocabulary)
- [Fine Grained](#Fine-Grained)
- [Long Tail](#Long-Tail)
- [Video Open World Papers](#Video-Open-World-Papers)
- [Anomaly Detection](#Anomaly-Detection)

## Introduction 
Let us consider a Telsa! Car is our running example during this repo. One day Telsa engineers decide to train their autopilot model. First, they meticulously collect these data on mornings on the way to their job while contemplating their life choices in the traffic. Then they give those videos to some people to categorize over several categories like cars, bikes, people, birds, and trees. They then train the model on the collected, labeled data, which produces a good performance on the evaluation data.

### Unseen Environments
The scientists put their model into action, and somebody uses their car at night. What the hell! Their model knows how a human looks in the day but not at night (thank god they still kept the mechanical brakes, so nobody hurts in this story). While these scientists have panicked and feared that they could get fired, a young scientist among them suggests that we can collect video data on our way back home at night while thinking about our following jobs. Sadly, they do not have enough money, so they can not afford to ask the bored annotators to categorize these videos again.

So they want to find a way to use their previously labeled day videos plus these unlabelled night videos to make a model that applies to night videos. The problem is that when they train a model in the day videos (which we call source data) while testing on night videos (which we call target data), there is a considerable distribution shift between source and target. Since computer vision models are functions that convert input distribution to output distribution, giving them a very different distribution does not produce the same output distribution. 
Thus we use labeled source data and unlabelled target data to make a model suitable to deal with the target distribution data at test time. This problem has been addressed in computer vision as **Domain Adaptation** and for video data as **Video Domain Adaptation**.
For a comprehensive list of papers see [Domain Adaptation](https://github.com/zhaoxin94/awesome-domain-adaptation) or [Video Domain Adaptation](https://github.com/xuyu0010/awesome-video-domain-adaptation).

Anyway, while hailing the young scientist for saving the product and their jobs, an unsatisfied customer calls to let them know that their model gets confused while crossing the bridges (this is why you should not rely on AI that much). This outcome is because these scientists lived in a city void of bridges, so their model gets confused when it encounters people on the bridges during test time. While panicking, team members are searching for the following job positions on LinkedIn; another young scientist calms the group and says, "What if we train a model on our training data that is guaranteed to work on an unseen target domain." Confused team members look at him and ask how to do that; one member says do you mean we consider the different parts of the city as separate domains and then learn a model which can be generalized from one domain to the other? He answers yes, and this problem has been addressed in literature as **Multi-source Domain Generalization**. Finally, a sleepy member of the group brings his head up and says, well, the problem then would be that a model that fails on bridges still can work well in different parts of cities. Before he goes back to his nap, members ask, " What do you suggest? And he says a model which only uses the source dataset in training but guarantees that it will work well on the target data. This problem is called **Single-source Domain Generalization** and for videos, **Video Domain Generalization**. 
For a comprehensive list of papers see [Domain Generalization](https://github.com/amber0309/Domain-generalization) and [Video Domain Generalization](https://github.com/thuml/VideoDG).

### Unseen Categories
While Telsa scientists are celebrating their success, they receive a call from their boss that for the next upgrade, they also need their cars to detect people's actions. Unfortunately, the company budget is low, so including all actions in the training data is infeasible. Hence they decide that instead of using all possible actions, they consider a subset of categories as *seen* actions while they count the rest as *unseen* actions while they try to find a set of attributes for these actions. This approach is called **Zero-Shot Learning** and its action version **Zero-Shot Action Recognition**. For a comprehensive list of papers, see [Zero-Shot Learning](https://github.com/sbharadwajj/awesome-zero-shot-learning) and [Zero-Shot Action Recognition](https://arxiv.org/abs/1909.06423).

While happy with the achievement that they could do well in unseen categories, they decided to bring their model to the real world. But unfortunately, these models became biased to either unseen or seen categories. So the more realistic problem is what we should do if we encounter both seen and unseen categories during test time. This problem is called **Generalized Zero-Shot Learning** or its action version **Generalized Zero-Shot Action Learning**. For more information, see [Generalized Zero-Shot Learning](https://arxiv.org/abs/2011.08641) and [Generalized Zero-Shot Action Learning](https://arxiv.org/abs/1710.07455).

### Unknown Categories
Telsa scientists are relieved that their autopilot model is ready to be employed in real world. Their model can handle unseen environments and unseen actions and categories. 


| **Problem Name**  |     **Problem Goal** |
|---------------|-----------|
| Zero-shot Learning| Classify test samples from a set of unseen but known categories.| 
| Generalized Zero-shot Learning | Classify test samples from a set of seen and unseen but known categories.| 
| Open-set Recognition  | Classify test samples from a set of seen categories while rejecting samples from unknown categories.|
| Novel Class Discovery | Classify test samples from a set of unknown categories into proposed clasters.|
| Generalized Category Discovery | Classify test samples from a set of seen or unknown categories into seen categories or proposed clasters.|
| Open vocabulary | Classify test samples from a set of seen or unknown categories into proposed clasters and find the corresponding name for that claster with the help of additional information like another modality or language models.|

<!-- #### CVPR
#### ICLR
#### NeurIPS
#### ICCV
#### ICML
#### IEEE-Access
#### ECCV
#### AAAI
#### TPAMI
#### CVPRw
#### WACV
#### IJCV
#### BMVC
#### ICCw -->
<!----------------------------------------------------------------------------------------------------------------------------------------------->
## Zero-Shot Learning
### 2023 Papers
### 2022 Papers
### 2021 Papers
<!-- #### CVPR
#### ICLR
#### NeurIPS
#### ICCV
#### ICML
#### IEEE-Access
#### ECCV
#### AAAI
#### TPAMI
#### CVPRw
#### WACV
#### IJCV
#### BMVC
#### ICCw
#### Arxiv & Others-->
<!----------------------------------------------------------------------------------------------------------------------------------------------->
## Out-of-Distribution Detection
### 2022 Papers
<!-- #### CVPR
#### ICLR
#### NeurIPS
#### ICCV-->
#### ICML
* Open-Sampling: Exploring Out-of-Distribution Data for Re-balancing long-tailed Datasets (ICML 2022) 
[[Paper](https://proceedings.mlr.press/v162/wei22c/wei22c.pdf)]
[[Code](https://github.com/hongxin001/logitnorm_ood)]<br>
*Datasets: long-tailed CIFAR10/100, CelebA-5, Places-LT*<br>
*Task: Out-of-Distribution detection*
<!-- #### IEEE-Access
#### ECCV
#### AAAI
#### TPAMI
#### CVPRw
#### WACV
#### IJCV
#### BMVC
#### ICCw -->
#### BMVC 
* OSM: An Open Set Matting Framework with OOD Detection and Few-Shot Matting (BMVC 2022) 
[[Paper]]
[[Code]]<br>
*Datasets:*<br>
*Task: OOD Detection*

<!----------------------------------------------------------------------------------------------------------------------------------------------->
## Open-Set Recognition 
### 2023 Papers
<!-- #### CVPR-->
#### ICLR
<!--#### NeurIPS
#### ICCV
#### ICML
#### IEEE-Access
#### ECCV
#### AAAI
#### TPAMI
#### CVPRw-->
#### WACV
<!--#### IJCV
#### BMVC
#### ICCw -->
### 2022 Papers
#### CVPR
* Expanding Low-Density Latent Regions for Open-Set Object Detection (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Han_Expanding_Low-Density_Latent_Regions_for_Open-Set_Object_Detection_CVPR_2022_paper.pdf)]
[[Code](https://github.com/csuhan/opendet2)]<br>
*Datasets: PASCAL VOC, MS COCO*<br> 
*Task: Object Detection* 
 
* Task-Adaptive Negative Envision for Few-Shot Open-Set Recognition (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Task-Adaptive_Negative_Envision_for_Few-Shot_Open-Set_Recognition_CVPR_2022_paper.pdf)]
[[Code](https://github.com/shiyuanh/TANE)]<br>
*Datasets: MiniImageNet, TieredImageNet*<br>
*Task: Few-Shot Open-Set Recognition*

* Active Learning for Open-Set Annotation (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Ning_Active_Learning_for_Open-Set_Annotation_CVPR_2022_paper.pdf)]<br>
*Datasets: CIFAR10, CIFAR100, TinyImageNet*<br>
*Task: Active Learning*

* SimT: Handling Open-set Noise for Domain Adaptive Semantic Segmentation (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_SimT_Handling_Open-Set_Noise_for_Domain_Adaptive_Semantic_Segmentation_CVPR_2022_paper.pdf)]
[[Code](https://github.com/CityU-AIM-Group/SimT)]<br>
*Datasets: GTA5→Cityscapes, Endovis17→Endovis18*<br>
*Task: Semantic Segmentation*

* Catching Both Gray and Black Swans: Open-set Supervised Anomaly Detection (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Ding_Catching_Both_Gray_and_Black_Swans_Open-Set_Supervised_Anomaly_Detection_CVPR_2022_paper.pdf)]
[[Code](https://github.com/choubo/DRA)]<br>
*Datasets: Hyper-Kvasir*<br>
*Task: Anomaly Detection*

* OSSGAN: Open-Set Semi-Supervised Image Generation (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Katsumata_OSSGAN_Open-Set_Semi-Supervised_Image_Generation_CVPR_2022_paper.pdf)]
[[Code](https://github.com/raven38/OSSGAN)]<br>
*Datasets: Tiny ImageNet, ImageNet ILSVRC2012*<br>
*Task: Image Generation*

* Open-Set Text Recognition via Character-Context Decoupling (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Open-Set_Text_Recognition_via_Character-Context_Decoupling_CVPR_2022_paper.pdf)]
[[Code](https://github.com/MisaOgura/flashtorch)]<br>
*Datasets: OSOCR (chinese train japanese test), HWDB and CTW*<br>
*Task: Text Recognition*

* OW-DETR: Open-world Detection Transformer (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Gupta_OW-DETR_Open-World_Detection_Transformer_CVPR_2022_paper.pdf)]
[[Code](https://github.com/akshitac8/OW-DETR)]<br>
*Datasets: MS-COCO, PascalVOC*<br>
*Task: Object Detection*

* ProposalCLIP: Unsupervised Open-Category Object Proposal Generationvia Exploiting CLIP Cues (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Shi_ProposalCLIP_Unsupervised_Open-Category_Object_Proposal_Generation_via_Exploiting_CLIP_Cues_CVPR_2022_paper.pdf)]
[[Code]]<br>
*Datasets:  PASCAL VOC 2007, COCO 2017, Visual Genome*<br>
*Task: Object Proposal Generation*

* SpaceEdit: Learning a Unified Editing Space for Open-Domain Image Color Editing (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Shi_SpaceEdit_Learning_a_Unified_Editing_Space_for_Open-Domain_Image_Color_CVPR_2022_paper.pdf)]
[[Code](https://jshi31.github.io/SpaceEdit)]<br>
*Datasets: Adobe Discover dataset, MA5K-Req*<br>
*Task: Image Editing*

* KG-SP: Knowledge Guided Simple Primitivesfor Open World Compositional Zero-Shot Learning (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Karthik_KG-SP_Knowledge_Guided_Simple_Primitives_for_Open_World_Compositional_Zero-Shot_CVPR_2022_paper.pdf)]
[[Code](https://github.com/ExplainableML/KG-SP)]<br>
*Datasets: UT-Zappos, MIT-States, C-GQA*<br>
*Task: Compositional Zero-Shot Learning*

* Safe-Student for Safe Deep Semi-Supervised Learning With Unseen-Class Unlabeled Data (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Safe-Student_for_Safe_Deep_Semi-Supervised_Learning_With_Unseen-Class_Unlabeled_Data_CVPR_2022_paper.pdf)]
[[Code]]<br>
*Datasets: MNIST; CIFAR-10; CIFAR-100; TinyImagenet*<br>
*Task: Image Classification*

* Open-Domain, Content-based, Multi-modal Fact-checking of Out-of-Context Images via Online Resources (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Abdelnabi_Open-Domain_Content-Based_Multi-Modal_Fact-Checking_of_Out-of-Context_Images_via_Online_Resources_CVPR_2022_paper.pdf)]
[[Code](https://s-abdelnabi.github.io/OoC-multi-modal-fc)]<br>
*Datasets: NewsCLIPpings*<br>
*Task: Multi-modal Fact-checking*


#### ICLR

* Open-set Recognition: A good closed-set classifier is all you need? (ICLR 2022 Oral) 
[[Paper](https://openreview.net/pdf?id=5hLP5JY9S2d)]
[[Code](https://github.com/sgvaze/osr_closed_set_all_you_need)]<br>
*Datasets:  ImageNet-21K-P, CUB, Stanford Car, FGVC-Aircraft, MNIST, SVHN, CIFAR10, CIFAR+N, TinyImageNet*<br>
*Task: Open-set recognition*

#### NeurIPS 
* Interpretable Open-Set Domain Adaptation via Angular Margin Separation	(NeurIPS 2022) 
[[Paper](https://arxiv.org/pdf/2202.01339.pdf)]
[[Code](https://github.com/sungnyun/understanding-cdfsl)]<br>
*Datasets: ImageNet, tieredImageNet, and miniImageNet for source domain similarity to ImageNet: Places,CUB,Cars,Plantae,EuroSAT,CropDisease,ISIC,ChestX*<br>
*Task: Active Learning*

* OpenAUC: Towards AUC-Oriented Open-Set Recognition	(NeurIPS 2022) 
[[Paper](https://arxiv.org/pdf/2210.13458v1.pdf)]
[[Code]]<br>
*Datasets: MNIST1, SVHN2, CIFAR10, CIFAR+10, CIFAR+50, TinyImageNet, CUB*<br>
*Task: Image Classification*

* Unknown-Aware Domain Adversarial Learning for Open-Set Domain Adaptation	(NeurIPS 2022) 
[[Paper](https://arxiv.org/pdf/2206.07551.pdf)]
[[Code](https://github.com/JoonHo-Jang/UADAL)]<br>
*Datasets: Office-31, Office-Home, VisDA*<br>
*Task: Domain Adaptation*

* Towards Open Set 3D Learning:Benchmarking and Understanding Semantic Novelty Detection on Point Clouds	(NeurIPS 2022) 
[[Paper](https://openreview.net/pdf?id=X2dHozbd1at)]
[[Code](https://github.com/antoalli/3D_OS)]<br>
*Datasets: ShapeNetCore v2,  ModelNet40->ScanObjectNN, ScanObjectNN*<br>
*Task: Point Cloud Novelty Detection*

<!-- #### ICCV
#### ICML
#### IEEE-Access -->
#### ECCV 
* Open-Set Semi-Supervised Object Detection (ECCV 2022 Oral) 
[[Paper](https://arxiv.org/pdf/2208.13722v1.pdf)]
[[Code]]<br>
*Datasets: COCO, OpenImages*<br>
*Task: Semi-Supervised Object Detection*

* Few-Shot Class-Incremental Learning from an Open-Set Perspective (ECCV 2022) 
[[Paper](https://arxiv.org/pdf/2208.00147v1.pdf)]
[[Code](https://github.com/CanPeng123/FSCIL_ALICE.git)]<br>
*Datasets: CIFAR100, miniImageNet, and CUB200*<br>
*Task: Image Classification*

* Towards Accurate Open-Set Recognition via Background-Class Regularization (ECCV 2022) 
[[Paper](https://arxiv.org/pdf/2207.10287v1.pdf)]
[[Code]]<br>
*Datasets: SVHN, CIFAR10 & CIFAR100, TinyImageNet*<br>
*Task: Image Classification*

* DenseHybrid: Hybrid Anomaly Detection for Dense Open-set Recognition (ECCV 2022) 
[[Paper](https://arxiv.org/pdf/2207.02606v1.pdf)]
[[Code](https://github.com/matejgrcic/DenseHybrid)]<br>
*Datasets: Fishyscapes, SegmentMeIfYouCan (SMIYC), StreetHazards*<br>
*Task: Anomaly Detection*

* Difficulty-Aware Simulator for Open Set Recognition (ECCV 2022) 
[[Paper](https://arxiv.org/pdf/2207.10024v1.pdf)]
[[Code](https://github.com/wjun0830/Difficulty-Aware-Simulator)]<br>
*Datasets: MNIST,SVHN,CIFAR10, CIFAR+10, CIFAR+50, Tiny-ImageNet, MNIST->Noise, MNIST Noise, Omniglot*<br>
*Task: Image Generation*

* Learning to Detect Every Thing in an Open World	(ECCV 2022) 
[[Paper](https://arxiv.org/pdf/2112.01698v2.pdf)]
[[Code](https://github.com/ksaito-ut/openworld_ldet)]<br>
*Datasets: COCO, Cityscapes, (test: UVO, Obj365, Mappilary Vista)*<br>
*Task: Object Detection*


* Open-world Semantic Segmentation via Contrasting and Clustering Vision-language Embedding (ECCV 2022) 
[[Paper](https://arxiv.org/pdf/2207.08455v2.pdf)]
[[Code]]<br>
*Datasets: PAS-CAL VOC, PASCAL Context, and COCO Stuff*<br>
*Task: Semantic Segmentation*

* Interpretable Open-Set Domain Adaptation via Angular Margin Separation	(ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136940001.pdf)]
[[Code](https://github.com/LeoXinhaoLee/AMS)]<br>
*Datasets: D2AwA, I2AwA*<br>
*Task: Domain Adaptation*

* PSS: Progressive Sample Selection for Open-World Visual Representation Learning	(ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136910269.pdf)]
[[Code](https://github.com/dmlc/dgl/tree/master/examples/pytorch/hilander/PSS)]<br>
*Datasets: iNaturalist, IMDB, Deep-Glint, IJB-C*<br>
*Task: Image retrieval*

* Fusing Local Similarities for Retrieval-based 3D Orientation Estimation of Unseen Objects	(ECCV 2022) (Open-set?)
[[Paper](https://arxiv.org/pdf/2203.08472v2.pdf)]
[[Code](https://sailor-z.github.io/projects/Unseen_Object_Pose.html)]<br>
*Datasets: LineMOD and LineMOD-O, T-LESS*<br>
*Task: 3D Oriention Estimation*

* Unknown-Oriented Learning for Open Set Domain Adaptation	(ECCV 2022)
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930328.pdf)]
[[Code]]<br>
*Datasets: SVHN→MNIST, MNIST→USPS and USPS→MNIST; Office-Home; Endo-c2k -> KID WCE*<br>
*Task: Domain Adaptation*

* UC-OWOD: Unknown-Classified Open World Object Detection	(ECCV 2022)
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700191.pdf)]
[[Code](https://github.com/JohnWuzh/UC-OWOD)]<br>
*Datasets: Pascal VOC; and MS-COCO*<br>
*Task: Object Detection*

* Open-world Semantic Segmentation for LIDAR Point Clouds	(ECCV 2022)
[[Paper](https://arxiv.org/pdf/2208.11113v1.pdf)]
[[Code](https://github.com/Jun-CEN/Open_world_3D_semantic_segmentation)]<br>
*Datasets: SemanticKITTI, nuScenes*<br>
*Task: Semantic Segmentation on LIDAR Point Clouds*

#### AAAI
* Learngene: From Open-World to Your Learning Task	(AAAI 2022)
[[Paper]]
[[Code]]<br>
*Datasets:*<br>
*Task:*

#### TPAMI
* Open Long-Tailed RecognitionIn A Dynamic World	(TPAMI 2022) 
[[Paper](https://arxiv.org/pdf/2208.08349v1.pdf)]
[[Code](https://liuziwei7.github.io/projects/LongTail.html)]<br>
*Datasets: CIFAR-10-LT,CIFAR-100-LT, and iNaturalist-18, Places-LT,  MS1M-LT, SUN-LT*<br>
*Task: Image Classification*

#### CVPRw

* Variable Few Shot Class Incremental and Open World Learning (CVPRw 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022W/CLVision/papers/Ahmad_Variable_Few_Shot_Class_Incremental_and_Open_World_Learning_CVPRW_2022_paper.pdf)]
[[Code](https://github.com/TouqeerAhmad/VFSOWL)]<br>
*Datasets: Caltech-UCSD Birds-200-2011 CUB200; miniImageNet*<br>
*Task: Image Classification*

* Towards Open-Set Object Detection and Discovery (CVPRw 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022W/L3D-IVU/papers/Zheng_Towards_Open-Set_Object_Detection_and_Discovery_CVPRW_2022_paper.pdf)]
[[Code]]<br>
*Datasets: Pascal VOC 2007; MS-COCO*<br>
*Task: Image Classification*

* Open-Set Domain Adaptation Under Few Source-Domain Labeled Samples (CVPRw 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022W/L3D-IVU/papers/Rakshit_Open-Set_Domain_Adaptation_Under_Few_Source-Domain_Labeled_Samples_CVPRW_2022_paper.pdf)]
[[Code]]<br>
*Datasets: Office-31; Mini-domainNet; NPU-RSDA*<br>
*Task: Image Classification*


#### WACV
* Few-Shot Open-Set Recognition of Hyperspectral Images With Outlier Calibration Network (WACV 2022) 
[[Paper](https://openaccess.thecvf.com/content/WACV2022/papers/Pal_Few-Shot_Open-Set_Recognition_of_Hyperspectral_Images_With_Outlier_Calibration_Network_WACV_2022_paper.pdf)]
[[Code](https://github.com/DebabrataPal7/OCN)]<br>
*Datasets:  Indian Pines(IP), Salinas, University of Pavia, Houston-2013*<br>
*Task: Hyperspectral Image Classification*

* Distance-based Hyperspherical Classification for Multi-source Open-Set Domain Adaptation	(WACV 2022) 
[[Paper](https://arxiv.org/pdf/2107.02067v3.pdf)]
[[Code](https://github.com/silvia1993/HyMOS)]<br>
*Datasets: Office-31, Office-Home, DomainNet*<br>
*Task: Domain Adaptation*

* SeeTek: Very Large-Scale Open-Set Logo Recognition With Text-Aware Metric Learning	(WACV 2022) 
[[Paper](https://openaccess.thecvf.com/content/WACV2022/papers/Li_SeeTek_Very_Large-Scale_Open-Set_Logo_Recognition_With_Text-Aware_Metric_Learning_WACV_2022_paper.pdf)]
[[Code]]<br>
*Datasets:*<br>
*Task:*

* Novel Ensemble Diversification Methods for Open-Set Scenarios	(WACV 2022) 
[[Paper](https://openaccess.thecvf.com/content/WACV2022/papers/Farber_Novel_Ensemble_Diversification_Methods_for_Open-Set_Scenarios_WACV_2022_paper.pdf)]
[[Code]]<br>
*Datasets:*<br>
*Task:*

* Learning To Generate the Unknowns as a Remedy to the Open-Set Domain Shift	(WACV 2022) 
[[Paper](https://openaccess.thecvf.com/content/WACV2022/papers/Baktashmotlagh_Learning_To_Generate_the_Unknowns_as_a_Remedy_to_the_WACV_2022_paper.pdf)]
[[Code]]<br>
*Datasets:*<br>
*Task:*


* Adversarial Open Domain Adaptation for Sketch-to-Photo Synthesis	(WACV 2022) 
[[Paper](https://arxiv.org/pdf/2104.05703v2.pdf)]
[[Code](https://github.com/Mukosame/AODA)]<br>
*Datasets:*<br>
*Task: Domain Adaptation*

<!-- #### IJCV -->

#### BMVC
* Dual Decision Improves Open-Set Panoptic Segmentation (BMVC 2022) 
[[Paper](https://arxiv.org/pdf/2207.02504v3.pdf)]
[[Code](https://github.com/jd730/EOPSN.git)]<br>
*Datasets: MS-COCO 2017*<br>
*Task: Panoptic Segmentation*

<!-- 
#### ICCVw
#### Arxiv & Others -->
### 2021 Papers
#### CVPR
* Counterfactual Zero-Shot and Open-Set Visual Recognition (CVPR 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yue_Counterfactual_Zero-Shot_and_Open-Set_Visual_Recognition_CVPR_2021_paper.pdf)]
[[Code]]<br>
*Datasets: MNIST, SVHN,CIFAR10 and CIFAR100*<br>
*Task: Object Detection*

* Towards Open World Object Detection (CVPR 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Joseph_Towards_Open_World_Object_Detection_CVPR_2021_paper.pdf)]
[[Code]]<br>
*Datasets:*<br>
*Task: Object Detection*

* Learning Placeholders for Open-Set Recognition (CVPR 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhou_Learning_Placeholders_for_Open-Set_Recognition_CVPR_2021_paper.pdf)]
[[Code]]<br>
*Datasets:*<br>
*Task: Object Detection*

* Few-Shot Open-Set Recognition by Transformation Consistency (CVPR 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Jeong_Few-Shot_Open-Set_Recognition_by_Transformation_Consistency_CVPR_2021_paper.pdf)]
[[Code]]<br>
*Datasets:*<br>
*Task: Object Detection*

* OpenMix: Reviving Known Knowledge for Discovering Novel Visual Categories in an Open World (CVPR 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhong_OpenMix_Reviving_Known_Knowledge_for_Discovering_Novel_Visual_Categories_in_CVPR_2021_paper.pdf)]
[[Code]]<br>
*Datasets: CIFAR10, CIFAR 100, Imagenet*<br>
*Task: Object Detection*
<!-- #### ICLR
#### NeurIPS
#### ICCV
#### ICML
#### IEEE-Access
#### ECCV
#### AAAI
#### TPAMI
#### CVPRw
#### WACV
#### IJCV
#### BMVC
#### ICCw
#### Arxiv & Others-->

<!----------------------------------------------------------------------------------------------------------------------------------------------->
## Novel Class Discovery
### 2023 Papers
<!-- #### CVPR
#### ICLR
#### NeurIPS
#### ICCV
#### ICML
#### IEEE-Access
#### ECCV
#### AAAI
#### TPAMI
#### CVPRw
#### WACV
#### IJCV
#### BMVC
#### ICCw
#### Arxiv & Others-->
### 2022 Papers
<!-- #### CVPR
#### ICLR
#### NeurIPS
#### ICCV
#### ICML
#### IEEE-Access
#### ECCV
#### AAAI
#### TPAMI
#### CVPRw
#### WACV
#### IJCV
#### BMVC
#### ICCw
#### Arxiv & Others-->
### 2021 Papers
#### CVPR
* OpenMix: Reviving Known Knowledge for Discovering Novel Visual Categories in an Open World (CVPR 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhong_OpenMix_Reviving_Known_Knowledge_for_Discovering_Novel_Visual_Categories_in_CVPR_2021_paper.pdf)]
[[Code]]<br>
*Datasets: CIFAR10, CIFAR 100, Imagenet*<br>
*Task: Object Detection*
<!--#### ICLR
#### NeurIPS
#### ICCV
#### ICML
#### IEEE-Access
#### ECCV
#### AAAI
#### TPAMI
#### CVPRw
#### WACV
#### IJCV
#### BMVC
#### ICCw
#### Arxiv & Others-->
<!----------------------------------------------------------------------------------------------------------------------------------------------->
## Open Vocabulary
### 2023 Papers
<!-- #### CVPR
#### ICLR
#### NeurIPS
#### ICCV
#### ICML
#### IEEE-Access
#### ECCV
#### AAAI
#### TPAMI
#### CVPRw
#### WACV
#### IJCV
#### BMVC
#### ICCw -->
### 2022 Papers
#### CVPR
* Open-Vocabulary One-Stage Detection with Hierarchical Visual-Language Knowledge Distillation	(CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Ma_Open-Vocabulary_One-Stage_Detection_With_Hierarchical_Visual-Language_Knowledge_Distillation_CVPR_2022_paper.pdf)]
[[Code](https://github.com/mengqiDyangge/HierKD)]<br>
*Datasets: MS COCO*<br>
*Task: Object Detection*

* Open-Vocabulary Instance Segmentation via Robust Cross-Modal Pseudo-Labeling	(CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Huynh_Open-Vocabulary_Instance_Segmentation_via_Robust_Cross-Modal_Pseudo-Labeling_CVPR_2022_paper.pdf)]
[[Code](https://github.com/hbdat/cvpr22_cross_modal_pseudo_labeling)]<br>
*Datasets: MS-COCO, Open Images, Conceptual Caption*<br>
*Task: Instance segmentation*

* Learning to Prompt for Open-Vocabulary Object Detection with Vision-Language Model (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Du_Learning_To_Prompt_for_Open-Vocabulary_Object_Detection_With_Vision-Language_Model_CVPR_2022_paper.pdf)]
[[Code](https://github.com/dyabel/detpro)]<br>
*Datasets: LVIS v1, Pascal  VOC  Dataset, COCO, Objects365  Dataset*<br>
*Task: Object detection and instance segmentation*



<!--#### ICLR
#### NeurIPS
#### ICCV
#### ICML
#### IEEE-Access-->
#### ECCV

* PromptDet: Towards Open-vocabulary Detection using Uncurated Images	(ECCV 2022) 
[[Paper](https://arxiv.org/pdf/2203.16513v2.pdf)]
[[Code](https://fcjian.github.io/promptdet)]<br>
*Datasets: LVIS, LAION-400M and LAION-Novel, COCO*<br>
*Task: Object Detection*

* Scaling Open-vocabulary Image Segmentation with Image-level Labels (ECCV 2022) 
[[Paper](https://arxiv.org/pdf/2112.12143v2.pdf)]
[[Code]]<br>
*Datasets: COCO, Localized Narrative (Loc. Narr.) test: PASCAL Context, PASCAL  VOC, ADE20k*<br>
*Task: Instance segmentation*

* Towards Open-vocabulary Scene Graph Generation with Prompt-based Finetuning (ECCV 2022) 
[[Paper](https://arxiv.org/pdf/2208.08165v3.pdf)]
[[Code]]<br>
*Datasets: Visual Genome(VG), GQA, Open-Image*<br>
*Task: Scene Graph Generation*

* Simple Open-Vocabulary Object Detection with Vision Transformers (ECCV 2022) 
[[Paper](https://arxiv.org/pdf/2205.06230v2.pdf)]
[[Code](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit)]<br>
*Datasets: OpenImages V4 (OI), Objects 365 (O365),and/or Visual Genome (VG) - Evaluation: COCO, LVIS, and O365*<br>
*Task: Object Detection*

* Open Vocabulary Object Detection with Pseudo Bounding-Box Labels (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700263.pdf)]
[[Code](https://github.com/salesforce/PB-OVD)]<br>
*Datasets: COCO Caption, Visual-Genome, and SBU Caption (Object names:  COCO,  PASCAL  VOC,  Objects365 and LVIS)*<br>
*Task: Object Detection*

* Open-Vocabulary DETR with Conditional Matching (ECCV 2022 Oral) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690107.pdf)]
[[Code](https://github.com/yuhangzang/OV-DETR)]<br>
*Datasets: LVIS, COCO*<br>
*Task: Object Detection*

* Improving Closed and Open-Vocabulary Attribute Prediction using Transformers (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850199.pdf)]
[[Code](https://vkhoi.github.io/TAP)]<br>
*Datasets: VAW (closed-set) LSA common, LSA common→rare, HICO*<br>
*Task: Attribute Prediction*

* A Simple Baseline for Open Vocabulary Semantic Segmentation with Pre-trained Vision-language Model (ECCV 2022) 
[[Paper]]
[[Code]]<br>
*Datasets: COCO Stuff; Pascal VOC 2012; Cityscapes; Pascal Context; ADE20K*<br>
*Task: Semantic Segmentation*

* A Dataset for Interactive Vision-Language Navigation with Unknown Command Feasibility (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680304.pdf)]
[[Code](https://github.com/aburns4/MoTIF)]<br>
*Datasets: MoTIF*<br>
*Task: Vision-Language Navigation (Apps)*

* Acknowledging the Unknown for Multi-label Learning with Single Positive Labels (ECCV 2022) 
[[Paper]]
[[Code]]<br>
*Datasets:*<br>
*Task:*

#### AAAI
* OVIS: Open-Vocabulary Visual Instance Search via Visual-Semantic Aligned Representation Learning (AAAI 2022) 
[[Paper](https://ojs.aaai.org/index.php/AAAI/article/download/20070/version/18367/19829)]
[[Code]]<br>
*Datasets:OVIS40; OVIS1400*<br>
*Task:*

<!--#### TPAMI
#### CVPRw
#### WACV
#### IJCV-->
#### BMVC
* Partially-Supervised Novel Object Captioning Using Context from Paired Data (BMVC 2022) 
[[Paper]]
[[Code]]<br>
*Datasets:*<br>
*Task:*

* Open-vocabulary Semantic Segmentation with Frozen Vision-Language Models (BMVC 2022) 
[[Paper]]
[[Code]]<br>
*Datasets:*<br>
*Task: Semantic Segmentation*
<!-- #### ICCw -->
#### Arxiv & Others
* Describing Sets of Images with Textual-PCA (EMNLP 2022) 
[[Paper](https://arxiv.org/pdf/2210.12112v1.pdf)]
[[Code](https://github.com/OdedH/textual-pca)]<br>
*Datasets: CelebA; Stanford Cars; COCO-Horses; LSUN-Church*<br>
*Task:*
### 2021 Papers
#### CVPR
* Open-Vocabulary Object Detection Using Captions	(CVPR 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zareian_Open-Vocabulary_Object_Detection_Using_Captions_CVPR_2021_paper.pdf)]
[[Code](https://github.com/alirezazareian/ovr-cnn)]<br>
*Datasets:*<br>
*Task:*
<!--#### ICLR
#### NeurIPS
#### ICCV
#### ICML
#### IEEE-Access
#### ECCV
#### AAAI
#### TPAMI
#### CVPRw
#### WACV
#### IJCV
#### BMVC
#### ICCw -->

<!----------------------------------------------------------------------------------------------------------------------------------------------->
## Fine Grained
### 2023 Papers
### 2022 Papers
### 2021 Papers
<!-- #### CVPR
#### ICLR
#### NeurIPS
#### ICCV
#### ICML
#### IEEE-Access
#### ECCV
#### AAAI
#### TPAMI
#### CVPRw
#### WACV
#### IJCV
#### BMVC
#### ICCw -->
<!----------------------------------------------------------------------------------------------------------------------------------------------->
## Long Tail
### 2023 Papers
### 2022 Papers
### 2021 Papers

#### TPAMI

* Open Long-Tailed RecognitionIn A Dynamic World	(TPAMI 2022) 
[[Paper](https://arxiv.org/pdf/2208.08349v1.pdf)]
[[Code](https://liuziwei7.github.io/projects/LongTail.html)]<br>
*Datasets: CIFAR-10-LT,CIFAR-100-LT, and iNaturalist-18, Places-LT,  MS1M-LT, SUN-LT*<br>
*Task: Image Classification*
<!----------------------------------------------------------------------------------------------------------------------------------------------->
# Video Open World Papers

<!----------------------------------------------------------------------------------------------------------------------------------------------->
## Anomaly Detection
### 2023 Papers
### 2022 Papers
### 2021 Papers
<!-- #### CVPR
#### ICLR
#### NeurIPS
#### ICCV
#### ICML
#### IEEE-Access
#### ECCV
#### AAAI
#### TPAMI
#### CVPRw
#### WACV
#### IJCV
#### BMVC
#### ICCw -->
