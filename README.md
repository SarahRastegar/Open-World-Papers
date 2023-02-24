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
- [Zero-Shot Learning Videos](#Zero-Shot-Learning-Videos)  
- [Out-of-Distribution Detection Videos](#Out-of-Distribution-Detection-Videos) 
- [Novel Class Discovery Videos](#Novel-Class-Discovery-Videos)
- [Open Vocabulary Videos](#Open-Vocabulary-Videos)
- [Fine Grained Videos](#Fine-Grained-Videos)
- [Long Tail Videos](#Long-Tail-Videos)
- [Anomaly Detection Videos](#Anomaly-Detection-Videos)

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
#### ICCw
#### Arxiv & Others-->

<!----------------------------------------------------------------------------------------------------------------------------------------------->
## Zero-Shot Learning
<!----------------------------------------------------------------------------------------------------------------------------------------------->

### 2023 Papers
<!-- #### CVPR-->
#### ICLR
* Relative representations enable zero-shot latent space communication (ICLR 2023 top 5%) 
[[Paper](https://openreview.net/forum?id=SrC-nwieGJ)]<br>
*Datasets: MNIST, F-MNIST, CIFAR-10, CIFAR-100, Cora, CiteSeer, PubMed, Amazon Reviews, TREC, DBpedia*<br>
*Task: Image Classification, Graph Node Classification, Image reconstruction, Text Classification*
<!--#### NeurIPS
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
#### CVPR
* KG-SP: Knowledge Guided Simple Primitivesfor Open World Compositional Zero-Shot Learning (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Karthik_KG-SP_Knowledge_Guided_Simple_Primitives_for_Open_World_Compositional_Zero-Shot_CVPR_2022_paper.pdf)]
[[Code](https://github.com/ExplainableML/KG-SP)]<br>
*Datasets: UT-Zappos, MIT-States, C-GQA*<br>
*Task: Compositional Zero-Shot Learning*

* Unseen Classes at a Later Time? No Problem (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Kuchibhotla_Unseen_Classes_at_a_Later_Time_No_Problem_CVPR_2022_paper.pdf)]<br>
*Datasets: AWA1  and  AWA2,  Attribute  Pascal  and  Yahoo(aPY), Caltech-UCSD-Birds 200-2011 (CUB) and SUN*<br>
*Task: Image Classification*

* Few-Shot Keypoint Detection With Uncertainty Learning for Unseen Species (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Lu_Few-Shot_Keypoint_Detection_With_Uncertainty_Learning_for_Unseen_Species_CVPR_2022_paper.pdf)]<br>
*Datasets: Animal  pose, CUB, NABird*<br>
*Task: Keypoint Detection*

* Distinguishing Unseen From Seen for Generalized Zero-Shot Learning (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Su_Distinguishing_Unseen_From_Seen_for_Generalized_Zero-Shot_Learning_CVPR_2022_paper.pdf)]<br>
*Datasets: Caltech-UCSD Birds-200-2011 (CUB), Ox-ford Flowers (FLO), SUN Attribute (SUN), Animals with Attributes 1 (AwA1) and Animals with Attributes 2(AwA2)*<br>
*Task: Image Classification*

* Siamese Contrastive Embedding Network for Compositional Zero-Shot Learning (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Siamese_Contrastive_Embedding_Network_for_Compositional_Zero-Shot_Learning_CVPR_2022_paper.pdf)]
[[Code](https://github.com/XDUxyLi/SCEN-master)]<br>
*Datasets: MIT-States, UT-Zappos, and C-GQA*<br>
*Task: Image Classification*

* ZeroCap: Zero-Shot Image-to-Text Generation for Visual-Semantic Arithmetic (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Tewel_ZeroCap_Zero-Shot_Image-to-Text_Generation_for_Visual-Semantic_Arithmetic_CVPR_2022_paper.pdf)]
[[Code](https://github.com/YoadTew/zero-shot-image-to-text)]<br>
*Datasets: COCO*<br>
*Task: Image Captioning*

* LiT: Zero-Shot Transfer With Locked-Image Text Tuning (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhai_LiT_Zero-Shot_Transfer_With_Locked-Image_Text_Tuning_CVPR_2022_paper.pdf)]<br>
*Datasets: CC12M; YFCC100m; ALIGN; ImageNet-v2, -R, -A, -ReaL, and ObjectNet, VTAB;  Cifar100; Pets; Wikipedia based Image Text (WIT)*<br>
*Task: Image-Text Retreival*

* Non-Generative Generalized Zero-Shot Learning via Task-Correlated Disentanglement and Controllable Samples Synthesis (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Feng_Non-Generative_Generalized_Zero-Shot_Learning_via_Task-Correlated_Disentanglement_and_Controllable_Samples_CVPR_2022_paper.pdf)]<br>
*Datasets: Animal with Attribute (AWA1), Animal with Attribute2 (AWA2), Caltech-UCSD Birds-200-2011(CUB), Oxford 102 flowers (FLO)*<br>
*Task: Image Classification*

* CLIP-Forge: Towards Zero-Shot Text-To-Shape Generation (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Sanghi_CLIP-Forge_Towards_Zero-Shot_Text-To-Shape_Generation_CVPR_2022_paper.pdf)]
[[Code](https://github.com/AutodeskAILab/Clip-Forge)]<br>
*Datasets: ShapeNet(v2) dataset*<br>
*Task: Text-To-Shape Generation*

* Zero-Shot Text-Guided Object Generation With Dream Fields (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Jain_Zero-Shot_Text-Guided_Object_Generation_With_Dream_Fields_CVPR_2022_paper.pdf)]
[[Code](https://ajayj.com/dreamfields)]<br>
*Datasets: COCO*<br>
*Task: Text-Guided Object Generation*

* En-Compactness: Self-Distillation Embedding & Contrastive Generation for Generalized Zero-Shot Learning (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Kong_En-Compactness_Self-Distillation_Embedding__Contrastive_Generation_for_Generalized_Zero-Shot_Learning_CVPR_2022_paper.pdf)]<br>
*Datasets: AWA1, AWA2, CUB, OxfordFlowers (FLO), Attributes  Pascal and Yahoo(APY)*<br>
*Task: Image Classification*

* VGSE: Visually-Grounded Semantic Embeddings for Zero-Shot Learning (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_VGSE_Visually-Grounded_Semantic_Embeddings_for_Zero-Shot_Learning_CVPR_2022_paper.pdf)]
[[Code](https://github.com/wenjiaXu/VGSE)]<br>
*Datasets: AWA2; CUB; SUN*<br>
*Task: Image Classification*

* Sketch3T: Test-Time Training for Zero-Shot SBIR (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Sain_Sketch3T_Test-Time_Training_for_Zero-Shot_SBIR_CVPR_2022_paper.pdf)]<br>
*Datasets: Sketchy;  TU-Berlin Extension*<br>
*Task: Sketch-Based Image Retrieval*

* MSDN: Mutually Semantic Distillation Network for Zero-Shot Learning (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_MSDN_Mutually_Semantic_Distillation_Network_for_Zero-Shot_Learning_CVPR_2022_paper.pdf)]
[[Code](https://anonymous.4open.science/r/MSDN)]<br>
*Datasets: CUB (Caltech  UCSD  Birds 200), SUN (SUN Attribute) and AWA2 (Animals with Attributes 2)*<br>
*Task: Image Classification*

* Decoupling Zero-Shot Semantic Segmentation (CVPR 2022) 
[[Paper](https://arxiv.org/pdf/2112.07910v2.pdf)]
[[Code](https://github.com/dingjiansw101/ZegFormer)]<br>
*Datasets: PASCAL VOC; COCO-Stuff*<br>
*Task: Semantic Segmentation*

* Robust Region Feature Synthesizer for Zero-Shot Object Detection (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Robust_Region_Feature_Synthesizer_for_Zero-Shot_Object_Detection_CVPR_2022_paper.pdf)]<br>
*Datasets: PASCAL VOC, COCO, and DIOR*<br>
*Task: Object Detection*

* IntraQ: Learning Synthetic Images With Intra-Class Heterogeneity for Zero-Shot Network Quantization (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhong_IntraQ_Learning_Synthetic_Images_With_Intra-Class_Heterogeneity_for_Zero-Shot_Network_CVPR_2022_paper.pdf)]
[[Code](https://github.com/zysxmu/IntraQ)]<br>
*Datasets: CIFAR-10/100; ImageNet*<br>
*Task: Zero-Shot Quantization*

* It's All in the Teacher: Zero-Shot Quantization Brought Closer to the Teacher (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Choi_Its_All_in_the_Teacher_Zero-Shot_Quantization_Brought_Closer_to_CVPR_2022_paper.pdf)]<br>
*Datasets: CIFAR-10/100; ImageNet*<br>
*Task: Zero-Shot Quantization*

* Robust Fine-Tuning of Zero-Shot Models (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.pdf)]<br>
*Datasets: ImageNet distribution shifts (ImageNetV2, ImageNet-R,ObjectNet, and ImageNet-A, ImageNet Sketch); CIFAR10.1 &10.2*<br>
*Task: Zero-Shot Distribution Shift Robustness*

* Neural Mean Discrepancy for Efficient Out-of-Distribution Detection (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Dong_Neural_Mean_Discrepancy_for_Efficient_Out-of-Distribution_Detection_CVPR_2022_paper.pdf)]<br>
*Datasets: CIFAR-10, CIFAR-100, SVHN, croppedImageNet,  cropped  LSUN,  iSUN,  and  Texture*<br>
*Task: Image Classification*

<!-- #### ICLR -->
#### NeurIPS
* Make an Omelette with Breaking Eggs: Zero-Shot Learning for Novel Attribute Synthesis (NeurIPS 2022) 
[[Paper](https://arxiv.org/pdf/2111.14182v5.pdf)]
[[Code](https://yuhsuanli.github.io/ZSLA)]<br>
*Datasets: Caltech-UCSD Birds-200-2011 (CUB Dataset), α-CLEVR*<br>
*Task: Image Classification*

<!-- #### ICCV
#### ICML
#### IEEE-Access -->
#### ECCV
* Zero-Shot Attribute Attacks on Fine-Grained Recognition Models (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650257.pdf)]<br>
*Datasets: Caltech-UCSD Birds-200-2011(CUB), Animal with Attributes (AWA2) and SUN Attribute (SUN)*<br>
*Task: Image Classification*

* Zero-Shot Learning for Reflection Removal of Single 360-Degree Image (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790523.pdf)]<br>
*Datasets: 30 test 360-degree images*<br>
*Task: Reflection Removal*

* Exploring Hierarchical Graph Representation for Large-Scale Zero-Shot Image Classification (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800108.pdf)]
[[Code](https://kaiyi.me/p/hgrnet.html)]<br>
*Datasets: ImageNet-21K-D (D for Directed Acyclic Graph)*<br>
*Task: Image Classification*

* Learning Invariant Visual Representations for Compositional Zero-Shot Learning (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840335.pdf)]
[[Code](https://github.com/PRIS-CV/IVR)]<br>
*Datasets: Mit-States; UT-Zappos50K; Clothing16K, and AO-CLEVr*<br>
*Task: Image Retrieval*

* 3D Compositional Zero-Shot Learning with DeCompositional Consensus (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880704.pdf)]<br>
*Datasets: Compositional PartNet (C-PartNet)*<br>
*Task: Compositional Zero-Shot Segmentation*

* Zero-Shot Category-Level Object Pose Estimation (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136990509.pdf)]
[[Code](https://github.com/applied-ai-lab/zero-shot-pose)]<br>
*Datasets: Common Objects in 3D (CO3D); PoseContrast*<br>
*Task: Object Pose Estimation*

<!-- #### AAAI
#### TPAMI-->
#### CVPRw
* Semantically Grounded Visual Embeddings for Zero-Shot Learning (CVPRw 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022W/MULA/papers/Nawaz_Semantically_Grounded_Visual_Embeddings_for_Zero-Shot_Learning_CVPRW_2022_paper.pdf)]<br>
*Datasets: CUB(312−d), AWA(85−d) and aPY(64−d); FLO*<br>
*Task: Semantic Embeddings*

* Zero-Shot Learning Using Multimodal Descriptions (CVPRw 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022W/L3D-IVU/papers/Mall_Zero-Shot_Learning_Using_Multimodal_Descriptions_CVPRW_2022_paper.pdf)]<br>
*Datasets: CUB-200-2011 (CUB), SUN attributes (SUN) and DeepFashion (DF)*<br>
*Task: Multimodal Zero-Shot*

<!-- #### WACV
#### IJCV
#### BMVC
#### ICCw
#### Arxiv & Others-->

### 2021 Papers
#### CVPR
* Counterfactual Zero-Shot and Open-Set Visual Recognition (CVPR 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yue_Counterfactual_Zero-Shot_and_Open-Set_Visual_Recognition_CVPR_2021_paper.pdf)]<br>
*Datasets: MNIST, SVHN,CIFAR10 and CIFAR100*<br>
*Task: Object Detection*
<!-- #### ICLR
#### NeurIPS-->
#### ICCV
* Prototypical Matching and Open Set Rejection for Zero-Shot Semantic Segmentation (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Prototypical_Matching_and_Open_Set_Rejection_for_Zero-Shot_Semantic_Segmentation_ICCV_2021_paper.pdf)]<br>
*Datasets: Pascal VOC 2012, Pascal Context*<br>
*Task: Semantic Segmentation*

<!--#### ICML
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

### Older Papers
* Enhancing the reliability of out-of-distribution image detection in neural networks (ICLR 2018) 
[[Paper](https://openreview.net/pdf?id=H1VGkIxRZ)]
[[Code](https://github.com/facebookresearch/odin)]<br>
*Datasets: CIFAR-10, CIFAR-100, ImageNet, iSUN*<br>
*Task: Out-of-Distribution Image Classification*

* A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks (NeurIPS 2018) 
[[Paper](https://proceedings.neurips.cc/paper/2018/file/abdeb6f575ac5c6676b747bca8d09cc2-Paper.pdf)]
[[Code](https://github.com/pokaxpoka/deep_Mahalanobis_detector)]<br>
*Datasets: CIFAR, SVHN, ImageNet and LSUN*<br>
*Task: Out-of-Distribution Image Classification*

<!----------------------------------------------------------------------------------------------------------------------------------------------->
## Out-of-Distribution Detection
<!----------------------------------------------------------------------------------------------------------------------------------------------->

### 2023 Papers
<!-- #### CVPR-->
#### ICLR
* Harnessing Out-Of-Distribution Examples via Augmenting Content and Style (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=boNyg20-JDm)]<br>
*Datasets:  SVHN, CIFAR10, LSUN, DTD, CUB, Flowers, Caltech, Dogs*<br>
*Task: Out-Of-Distribution Detection*
<!--#### NeurIPS
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
### 2022 Papers
#### CVPR
* OoD-Bench: Quantifying and Understanding Two Dimensions of Out-of-Distribution Generalization (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Ye_OoD-Bench_Quantifying_and_Understanding_Two_Dimensions_of_Out-of-Distribution_Generalization_CVPR_2022_paper.pdf)]
[[Code](https://github.com/ynysjtu/ood_bench)]<br>
*Datasets: PACS; Office Home; TerraInc; Camelyon17; Colored MNIST; NICO; CelebA*<br>
*Task: Image Classification*

* Evading the Simplicity Bias: Training a Diverse Set of Models Discovers Solutions With Superior OOD Generalization (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Teney_Evading_the_Simplicity_Bias_Training_a_Diverse_Set_of_Models_CVPR_2022_paper.pdf)]
[[Code](https://github.com/dteney/collages-dataset)]<br>
*Datasets: [MNIST;CIFAR;  Fashion-MNIST; SVHN]  Biased activity recognition (BAR); PACS*<br>
*Task: Image Classification*

* Weakly Supervised Semantic Segmentation Using Out-of-Distribution Data (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_Weakly_Supervised_Semantic_Segmentation_Using_Out-of-Distribution_Data_CVPR_2022_paper.pdf)]
[[Code](https://github.com/naver-ai/w-ood)]<br>
*Datasets: Pascal VOC 2012, OpenImages, hard OoD dataset*<br>
*Task: Semantic Segmentation*

* DeepFace-EMD: Re-Ranking Using Patch-Wise Earth Mover's Distance Improves Out-of-Distribution Face Identification (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Phan_DeepFace-EMD_Re-Ranking_Using_Patch-Wise_Earth_Movers_Distance_Improves_Out-of-Distribution_Face_CVPR_2022_paper.pdf)]
[[Code](https://github.com/anguyen8/deepface-emd)]<br>
*Datasets: LFW, LFW-crop*<br>
*Task: Face Identification*

* Neural Mean Discrepancy for Efficient Out-of-Distribution Detection (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Dong_Neural_Mean_Discrepancy_for_Efficient_Out-of-Distribution_Detection_CVPR_2022_paper.pdf)]<br>
*Datasets: CIFAR-10, CIFAR-100, SVHN, croppedImageNet,  cropped  LSUN,  iSUN,  and  Texture*<br>
*Task: Image Classification*

* Deep Hybrid Models for Out-of-Distribution Detection (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Cao_Deep_Hybrid_Models_for_Out-of-Distribution_Detection_CVPR_2022_paper.pdf)]<br>
*Datasets: CIFAR-10 and CIFAR-100, SVHN, CLINC150*<br>
*Task: Image Classification*

* Amodal Segmentation Through Out-of-Task and Out-of-Distribution Generalization With a Bayesian Model (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Sun_Amodal_Segmentation_Through_Out-of-Task_and_Out-of-Distribution_Generalization_With_a_Bayesian_CVPR_2022_paper.pdf)]
[[Code](https://github.com/anonymous-submission-vision/Amodal-Bayesian)]<br>
*Datasets: OccludedVehicles; KINS; COCOA-cls*<br>
*Task: Instance Segmentation*

* Rethinking Reconstruction Autoencoder-Based Out-of-Distribution Detection (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Rethinking_Reconstruction_Autoencoder-Based_Out-of-Distribution_Detection_CVPR_2022_paper.pdf)]<br>
*Datasets: CIFAR10; CIFAR100*<br>
*Task: Image Classification*

* ViM: Out-of-Distribution With Virtual-Logit Matching (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_ViM_Out-of-Distribution_With_Virtual-Logit_Matching_CVPR_2022_paper.pdf)]
[[Code](https://github.com/haoqiwang/vim)]<br>
*Datasets: OpenImage-O; Texture; iNaturalist, ImageNet-O*<br>
*Task: Image Classification*

* Out-of-Distribution Generalization With Causal Invariant Transformations (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Out-of-Distribution_Generalization_With_Causal_Invariant_Transformations_CVPR_2022_paper.pdf)]
[[Code](https://www.mindspore.cn)]<br>
*Datasets: PACS; VLCS {VOC2007, LabelMe, Caltech101, SUN09}*<br>
*Task: Image Classification*

* Trustworthy Long-Tailed Classification (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Trustworthy_Long-Tailed_Classification_CVPR_2022_paper.pdf)]<br>
*Datasets: (CIFAR-10-LT, CIFAR-100-LT and ImageNet-LT) and three balanced OOD datasets (SVHN, ImageNet-open and Places-open)*<br>
*Task: Image Classification*

<!--#### ICLR
#### NeurIPS
#### ICCV-->
#### ICML
* Open-Sampling: Exploring Out-of-Distribution Data for Re-balancing long-tailed Datasets (ICML 2022) 
[[Paper](https://proceedings.mlr.press/v162/wei22c/wei22c.pdf)]
[[Code](https://github.com/hongxin001/logitnorm_ood)]<br>
*Datasets: long-tailed CIFAR10/100, CelebA-5, Places-LT*<br>
*Task: Image Classification*

<!-- #### IEEE-Access-->
#### ECCV

* OOD-CV: A Benchmark for Robustness to Out-of-Distribution Shifts of Individual Nuisances in Natural Images (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680158.pdf)]<br>
*Datasets: PASCAL3D+; OOD-CV*<br>
*Task: Image Classification, Object Detection, and 3D Pose Estimation*

* Out-of-Distribution Identification: Let Detector Tell Which I Am Not Sure (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700631.pdf)]<br>
*Datasets: PASCAL VOC-IO; Crack  Defect*<br>
*Task: Image Classification*

* Out-of-Distribution Detection with Boundary Aware Learning (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840232.pdf)]<br>
*Datasets: CIFAR-10 and CIFAR-100; SVHN and LSUN; TinyImageNet; MNIST; Fashion-MNIST; Omniglot*<br>
*Task: Image Classification*

* Out-of-Distribution Detection with Semantic Mismatch under Masking (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840369.pdf)]
[[Code](https://github.com/cure-lab/MOODCat)]<br>
*Datasets: Cifar-10, Cifar-100, SVHN, Texture, Places365, Lsun and Tiny-ImageNet*<br>
*Task: Image Classification*

* DICE: Leveraging Sparsification for Out-of-Distribution Detection (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840680.pdf)]
[[Code](https://github.com/deeplearning-wisc/dice.git)]<br>
*Datasets: CIFAR10; CIFAR100; Places365; Textures; iNaturalist; and SUN*<br>
*Task: Image Classification*

* Class Is Invariant to Context and Vice Versa: On Learning Invariance for Out-of-Distribution Generalization (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850089.pdf)]
[[Code](https://github.com/simpleshinobu/IRMCon)]<br>
*Datasets: Colored MNIST; Corrupted CIFAR-10, Biased Action Recognition(BAR); PACS*<br>
*Task: Image Classification*

* Data Invariants to Understand Unsupervised Out-of-Distribution Detection (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136910129.pdf)]<br>
*Datasets: CIFAR10; MVTec; SVHN; CIFAR100; DomainNet; coherence tomography (OCT); chest X-ray*<br>
*Task: Image Classification*

* Embedding Contrastive Unsupervised Features to Cluster in- and Out-of-Distribution Noise in Corrupted Image Datasets (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136910389.pdf)]
[[Code](https://github.com/PaulAlbert31/SNCF)]<br>
*Datasets: (mini) Webvision*<br>
*Task: Image Classification*

#### AAAI
* Gradient-Based Novelty Detection Boosted by Self-Supervised Binary Classification (AAAI 2022) 
[[Paper](https://ojs.aaai.org/index.php/AAAI/article/download/20812/version/19109/20571)]<br>
*Datasets: CIFAR-10, CIFAR-100, SVHN and TinyImageNet*<br>
*Task: Image Classification*

<!-- #### TPAMI-->
#### CVPRw
* Out-of-Distribution Detection in Unsupervised Continual Learning (CVPRw 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022W/CLVision/papers/He_Out-of-Distribution_Detection_in_Unsupervised_Continual_Learning_CVPRW_2022_paper.pdf)]<br>
*Datasets: CIFAR-100*<br>
*Task: Image Classification*

* Continual Learning Based on OOD Detection and Task Masking (CVPRw 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022W/CLVision/papers/Kim_Continual_Learning_Based_on_OOD_Detection_and_Task_Masking_CVPRW_2022_paper.pdf)]
[[Code](https://github.com/k-gyuhak/CLOM)]<br>
*Datasets: MNIST-5T; CIFAR10-5T; CIFAR100-10T; CIFAR100-20T; T-ImageNet-5T; T-ImageNet-10T*<br>
*Task: Image Classification*

* Class-Wise Thresholding for Robust Out-of-Distribution Detection (CVPRw 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022W/FaDE-TCV/papers/Guarrera_Class-Wise_Thresholding_for_Robust_Out-of-Distribution_Detection_CVPRW_2022_paper.pdf)]<br>
*Datasets: Places365; SVHN; German Traffic Sign Recognition Benchmark (GTSRB); ImageNet; Anime Faces; Fishes; Fruits; iSUN; Jig-saw Training; LSUN; Office; PACS; Texture*<br>
*Task: Image Classification*

* PyTorch-OOD: A Library for Out-of-Distribution Detection Based on PyTorch (CVPRw 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022W/HCIS/papers/Kirchheim_PyTorch-OOD_A_Library_for_Out-of-Distribution_Detection_Based_on_PyTorch_CVPRW_2022_paper.pdf)]
[[Code](https://gitlab.com/kkirchheim/pytorch-ood)]<br>
*Datasets: CIFAR 10 or CIFAR 100;  ImageNet-A; ImageNet-O; Newsgroups; ImageNet-R*<br>
*Task: Image Classification*

* RODD: A Self-Supervised Approach for Robust Out-of-Distribution Detection (CVPRw 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022W/ArtOfRobust/papers/Khalid_RODD_A_Self-Supervised_Approach_for_Robust_Out-of-Distribution_Detection_CVPRW_2022_paper.pdf)]
[[Code](https://github.com/UmarKhalidcs/RODD)]<br>
*Datasets: CIFAR-10 and CIFAR-100 as ID datasets and 7 OOD datasets. OOD datasets utilized are TinyImageNet-crop (TINc), TinyImageNet-resize(TINr), LSUN-resize (LSUN-r), Places, Textures, SVHN and iSUN*<br>
*Task: Image Classification*

<!--#### WACV
#### IJCV
#### BMVC
#### ICCw -->
#### BMVC 
* OSM: An Open Set Matting Framework with OOD Detection and Few-Shot Matting (BMVC 2022) 
[[Paper](https://bmvc2022.mpi-inf.mpg.de/0092.pdf)]<br>
*Datasets: SIMD*<br>
*Task: Out-of-Distribution Detection, Semantic Image Matting*
<!----------------------------------------------------------------------------------------------------------------------------------------------->
### 2021 Papers
<!-- #### CVPR
#### ICLR
#### NeurIPS-->
#### ICCV
* Trash To Treasure: Harvesting OOD Data With Cross-Modal Matching for Open-Set Semi-Supervised Learning (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Huang_Trash_To_Treasure_Harvesting_OOD_Data_With_Cross-Modal_Matching_for_ICCV_2021_paper.pdf)]<br>
*Datasets: CIFAR-10, Animal-10, Tiny-ImageNet, CIFAR100*<br>
*Task: Image Classification*

<!--#### ICML
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
## Open-Set Recognition 
<!----------------------------------------------------------------------------------------------------------------------------------------------->
### Surveys
* Learning and the Unknown: Surveying Steps toward Open World Recognition (AAAI 2019) 
[[Paper](https://ojs.aaai.org/index.php/AAAI/article/download/5054/4927)]

* Recent advances in open set recognition: A survey (TPAMI 2020) 
[[Paper](https://arxiv.org/abs/1811.08581v4)]

<!----------------------------------------------------------------------------------------------------------------------------------------------->
### 2023 Papers
<!-- #### CVPR-->
#### ICLR
* Towards Open Temporal Graph Neural Networks (ICLR 2023 top 5%) 
[[Paper](https://openreview.net/forum?id=N9Pk5iSCzAn)]<br>
*Datasets: Reddit, Yelp, Taobao*<br>
*Task: Graph Neural Networks*

* Adaptive Robust Evidential Optimization For Open Set Detection from Imbalanced Data (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=3yJ-hcJBqe)]<br>
*Datasets: CIFAR10, CIFAR100, ImageNet, MNIST, and Architecture Heritage Elements Dataset (AHED)*<br>
*Task: Image Classification*

* The Devil is in the Wrongly-classified Samples: Towards Unified Open-set Recognition (ICLR 2023) 
[[Paper](https://openreview.net/pdf?id=xLr0I_xYGAs)]<br>
*Datasets: CIFAR100, LSUN, MiTv2, UCF101, HMDB51*<br>
*Task: Image and Video Classification*

* Towards Addressing Label Skews in One-Shot Federated Learning (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=rzrqh85f4Sc)]
[[Code](https://github.com/Xtra-Computing/FedOV)]<br>
*Datasets: MNIST, Fashion-MNIST, CIFAR-10 and SVHN*<br>
*Task: Federated Learning*

* Evidential Uncertainty and Diversity Guided Active Learning for Scene Graph Generation (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=xI1ZTtVOtlz)]<br>
*Datasets: VG150*<br>
*Task: Scene Graph Generation*

* GOOD: Exploring geometric cues for detecting objects in an open world (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=W-nZDQyuy8D)]
[[Code](https://github.com/autonomousvision/good)]<br>
*Datasets: COCO*<br>
*Task: Object Detection*

* Unicom: Universal and Compact Representation Learning for Image Retrieval (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=3YFDsSRSxB-)]<br>
*Datasets: LAION 400M*<br>
*Task: Image Retrieval*

<!--#### NeurIPS
#### ICCV
#### ICML
#### IEEE-Access
#### ECCV
#### AAAI
#### TPAMI
#### CVPRw-->
#### WACV
* MORGAN: Meta-Learning-Based Few-Shot Open-Set Recognition via Generative Adversarial Network (WACV 2023) 
[[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Pal_MORGAN_Meta-Learning-Based_Few-Shot_Open-Set_Recognition_via_Generative_Adversarial_Network_WACV_2023_paper.pdf)]
[[Code](https://github.com/DebabrataPal7/MORGAN)]<br>
*Datasets: Indian  Pines (IP), Salinas, University of Pavia, Houston-2013*<br>
*Task: Hyper-Spectral Images*

* Ancestor Search: Generalized Open Set Recognition via Hyperbolic Side Information Learning (WACV 2023) 
[[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Dengxiong_Ancestor_Search_Generalized_Open_Set_Recognition_via_Hyperbolic_Side_Information_WACV_2023_paper.pdf)]<br>
*Datasets: CUB-200, AWA2, MNIST, CIFAR-10, CIFAR-100, SVHN, Tiny Imagenet*<br>
*Task: Image Classification*

* Large-Scale Open-Set Classification Protocols for ImageNet (WACV 2023) 
[[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Palechor_Large-Scale_Open-Set_Classification_Protocols_for_ImageNet_WACV_2023_paper.pdf)]
[[Code](https://github.com/AIML-IfI/openset-imagenet)]<br>
*Datasets: ILSVRC 2012*<br>
*Task: Image Classification*

* Contrastive Learning of Semantic Concepts for Open-Set Cross-Domain Retrieval (WACV 2023) 
[[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Agarwal_Contrastive_Learning_of_Semantic_Concepts_for_Open-Set_Cross-Domain_Retrieval_WACV_2023_paper.pdf)]<br>
*Datasets: DomainNet, PACS, Sketchy Extended*<br>
*Task: Open-Set Domain Generalization*

<!--#### IJCV
#### BMVC
#### ICCw -->
#### Arxiv & Others 
* OpenCon: Open-world Contrastive Learning (TMLR 2023) 
[[Paper](https://openreview.net/forum?id=2wWJxtpFer)]
[[Code](https://github.com/deeplearning-wisc/opencon)]<br>
*Datasets: CIFAR-10, CIFAR-100, Imagenet-100*<br>
*Task: Image Classification and Domain Adaptation*

<!----------------------------------------------------------------------------------------------------------------------------------------------->
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
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Shi_ProposalCLIP_Unsupervised_Open-Category_Object_Proposal_Generation_via_Exploiting_CLIP_Cues_CVPR_2022_paper.pdf)]<br>
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
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Safe-Student_for_Safe_Deep_Semi-Supervised_Learning_With_Unseen-Class_Unlabeled_Data_CVPR_2022_paper.pdf)]<br>
*Datasets: MNIST; CIFAR-10; CIFAR-100; TinyImagenet*<br>
*Task: Image Classification*

* Open-Domain, Content-based, Multi-modal Fact-checking of Out-of-Context Images via Online Resources (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Abdelnabi_Open-Domain_Content-Based_Multi-Modal_Fact-Checking_of_Out-of-Context_Images_via_Online_Resources_CVPR_2022_paper.pdf)]
[[Code](https://s-abdelnabi.github.io/OoC-multi-modal-fc)]<br>
*Datasets: NewsCLIPpings*<br>
*Task: Multimodal Fact-Checking*

#### ICLR

* Open-set Recognition: A good closed-set classifier is all you need? (ICLR 2022 Oral) 
[[Paper](https://openreview.net/pdf?id=5hLP5JY9S2d)]
[[Code](https://github.com/sgvaze/osr_closed_set_all_you_need)]<br>
*Datasets: ImageNet-21K-P, CUB, Stanford Car, FGVC-Aircraft, MNIST, SVHN, CIFAR10, CIFAR+N, TinyImageNet*<br>
*Task: Open-set Recognition*

* CrossMatch: Cross-Classifier Consistency Regularization for Open-Set Single Domain Generalization (ICLR 2022) 
[[Paper](https://openreview.net/forum?id=48RBsJwGkJf)]<br>
*Datasets: MNIST, SVHN,USPS, MNIST-M, SYN, Office31, Office-Home, PACS*<br>
*Task: Domain Generalization*

* Objects in Semantic Topology (ICLR 2022) 
[[Paper](https://openreview.net/forum?id=d5SCUJ5t1k)]<br>
*Datasets: Pascal VOC, MS-COCO*<br>
*Task: Object Detection*

#### NeurIPS 
* Rethinking Knowledge Graph Evaluation Under the Open-World Assumption (NeurIPS 2022 Oral) 
[[Paper](https://openreview.net/forum?id=5xiLuNutzJG)]
[[Code](https://github.com/GraphPKU/Open-World-KG)]<br>
*Datasets: family tree KG*<br>
*Task: Knowledge  Graph  Completion*

* Interpretable Open-Set Domain Adaptation via Angular Margin Separation	(NeurIPS 2022) 
[[Paper](https://arxiv.org/pdf/2202.01339.pdf)]
[[Code](https://github.com/sungnyun/understanding-cdfsl)]<br>
*Datasets: ImageNet, tieredImageNet, and miniImageNet for source domain similarity to ImageNet: Places,CUB,Cars,Plantae,EuroSAT,CropDisease,ISIC,ChestX*<br>
*Task: Active Learning*

* OpenAUC: Towards AUC-Oriented Open-Set Recognition	(NeurIPS 2022) 
[[Paper](https://arxiv.org/pdf/2210.13458v1.pdf)]<br>
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

* Domain Adaptation under Open Set Label Shift (NeurIPS 2022) 
[[Paper](https://openreview.net/forum?id=OMZG4vsKmm7)]
[[Code](https://github.com/acmi-lab/Open-Set-Label-Shift)]<br>
*Datasets: CIFAR10, CIFAR100, Entity30, Newsgroups-20, Tabula Muris, Dermnet (skin disease prediction), BreakHis(tumor cell classification)*<br>
*Task: Domain Adaptation*

* Meta-Query-Net: Resolving Purity-Informativeness Dilemma in Open-set Active Learning (NeurIPS 2022) 
[[Paper](https://openreview.net/forum?id=_kZVnosHbV3)]<br>
*Datasets: CIFAR10, CIFAR100, and ImageNet,  LSUN, Places365*<br>
*Task: Active Learning*

* Maximum Class Separation as Inductive Bias in One Matrix (NeurIPS 2022) 
[[Paper](https://openreview.net/forum?id=MbVS6BuJ3ql)]
[[Code](https://github.com/tkasarla/max-separation-as-inductive-bias)]<br>
*Datasets: CIFAR10, CIFAR100, and ImageNet*<br>
*Task: Image Classification*

* GlanceNets: Interpretable, Leak-proof Concept-based Models (NeurIPS 2022) 
[[Paper](https://openreview.net/forum?id=MbVS6BuJ3ql)]
[[Code](https://github.com/ema-marconato/glancenet)]<br>
*Datasets: dSprites, MPI3D, CelebA, even and odd MNIST images*<br>
*Task: Concept Leakage*

* Attracting and Dispersing: A Simple Approach for Source-free Domain Adaptation (NeurIPS 2022) 
[[Paper](https://openreview.net/forum?id=ZlCpRiZN7n)]
[[Code](https://github.com/Albert0147/AaD_SFDA)]<br>
*Datasets: Office-31,Office-Home and VisDA-C 2017*<br>
*Task: Domain Adaptation*

<!-- #### ICCV
#### ICML
#### IEEE-Access -->
#### ECCV 
* Open-Set Semi-Supervised Object Detection (ECCV 2022 Oral) 
[[Paper](https://arxiv.org/pdf/2208.13722v1.pdf)]<br>
*Datasets: COCO, OpenImages*<br>
*Task: Semi-Supervised Object Detection*

* Few-Shot Class-Incremental Learning from an Open-Set Perspective (ECCV 2022) 
[[Paper](https://arxiv.org/pdf/2208.00147v1.pdf)]
[[Code](https://github.com/CanPeng123/FSCIL_ALICE.git)]<br>
*Datasets: CIFAR100, miniImageNet, and CUB200*<br>
*Task: Image Classification*

* Towards Accurate Open-Set Recognition via Background-Class Regularization (ECCV 2022) 
[[Paper](https://arxiv.org/pdf/2207.10287v1.pdf)]<br>
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
[[Paper](https://arxiv.org/pdf/2207.08455v2.pdf)]<br>
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
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930328.pdf)]<br>
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
[[Paper](https://arxiv.org/abs/2106.06788v3)]<br>
*Datasets:CIFAR100, ImageNet100*<br>
*Task: Meta Learning*

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
[[Paper](https://openaccess.thecvf.com/content/CVPR2022W/L3D-IVU/papers/Zheng_Towards_Open-Set_Object_Detection_and_Discovery_CVPRW_2022_paper.pdf)]<br>
*Datasets: Pascal VOC 2007; MS-COCO*<br>
*Task: Image Classification*

* Open-Set Domain Adaptation Under Few Source-Domain Labeled Samples (CVPRw 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022W/L3D-IVU/papers/Rakshit_Open-Set_Domain_Adaptation_Under_Few_Source-Domain_Labeled_Samples_CVPRW_2022_paper.pdf)]<br>
*Datasets: Office-31; Mini-domainNet; NPU-RSDA*<br>
*Task: Image Classification*


#### WACV
* Few-Shot Open-Set Recognition of Hyperspectral Images With Outlier Calibration Network (WACV 2022) 
[[Paper](https://openaccess.thecvf.com/content/WACV2022/papers/Pal_Few-Shot_Open-Set_Recognition_of_Hyperspectral_Images_With_Outlier_Calibration_Network_WACV_2022_paper.pdf)]
[[Code](https://github.com/DebabrataPal7/OCN)]<br>
*Datasets:  Indian Pines(IP), Salinas, University of Pavia, Houston-2013*<br>
*Task: Hyperspectral Image Classification*

* Distance-based Hyperspherical Classification for Multi-source Open-Set Domain Adaptation (WACV 2022) 
[[Paper](https://arxiv.org/pdf/2107.02067v3.pdf)]
[[Code](https://github.com/silvia1993/HyMOS)]<br>
*Datasets: Office-31, Office-Home, DomainNet*<br>
*Task: Domain Adaptation*

* SeeTek: Very Large-Scale Open-Set Logo Recognition With Text-Aware Metric Learning (WACV 2022) 
[[Paper](https://openaccess.thecvf.com/content/WACV2022/papers/Li_SeeTek_Very_Large-Scale_Open-Set_Logo_Recognition_With_Text-Aware_Metric_Learning_WACV_2022_paper.pdf)]<br>
*Datasets: PL8K,LogoDet-3K, FlickrLogos-47, Logos-in-the-Wild (LitW), OpenLogo, BelgaLogos, Hard Evaluation dataset*<br>
*Task: Logo Recognition*

* Novel Ensemble Diversification Methods for Open-Set Scenarios	(WACV 2022) 
[[Paper](https://openaccess.thecvf.com/content/WACV2022/papers/Farber_Novel_Ensemble_Diversification_Methods_for_Open-Set_Scenarios_WACV_2022_paper.pdf)]<br>
*Datasets: Market-1501 (ReID), MS1MV2,  Labeled Faces in the Wild (LFW) (Face Recognition), CIFAR-100, CIFAR-10, TinyImageNet(crop) (OSR, OOD), SVHN and MNIST (OSR)*<br>
*Task: Ensemble Diversification*

* Learning To Generate the Unknowns as a Remedy to the Open-Set Domain Shift (WACV 2022) 
[[Paper](https://openaccess.thecvf.com/content/WACV2022/papers/Baktashmotlagh_Learning_To_Generate_the_Unknowns_as_a_Remedy_to_the_WACV_2022_paper.pdf)]<br>
*Datasets: Office-Home, VisDA-17, Syn2Real-O*<br>
*Task: Domain Adaptation*

* Adversarial Open Domain Adaptation for Sketch-to-Photo Synthesis (WACV 2022) 
[[Paper](https://arxiv.org/pdf/2104.05703v2.pdf)]
[[Code](https://github.com/Mukosame/AODA)]<br>
*Datasets: Scribble, SketchyCOCO*<br>
*Task: Domain Adaptation*

<!-- #### IJCV -->

#### BMVC
* Dual Decision Improves Open-Set Panoptic Segmentation (BMVC 2022) 
[[Paper](https://arxiv.org/pdf/2207.02504v3.pdf)]
[[Code](https://github.com/jd730/EOPSN.git)]<br>
*Datasets: MS-COCO 2017*<br>
*Task: Panoptic Segmentation*

<!-- 
#### ICCVw -->
#### Arxiv & Others 
* Open Set Domain Adaptation By Novel Class Discovery (ICME 2022) 
[[Paper](https://arxiv.org/abs/2203.03329v1)]<br>
*Datasets: Office-31, Office-Home, DomainNet*<br>
*Task: Image Classification and Domain Adaptation*
<!----------------------------------------------------------------------------------------------------------------------------------------------->
### 2021 Papers
#### CVPR
* Counterfactual Zero-Shot and Open-Set Visual Recognition (CVPR 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yue_Counterfactual_Zero-Shot_and_Open-Set_Visual_Recognition_CVPR_2021_paper.pdf)]<br>
*Datasets: MNIST, SVHN,CIFAR10 and CIFAR100*<br>
*Task: Image Classification*

* Towards Open World Object Detection (CVPR 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Joseph_Towards_Open_World_Object_Detection_CVPR_2021_paper.pdf)]
[[Code](https://github.com/JosephKJ/OWOD)]<br>
*Datasets: Pascal VOC, MS-COCO<br>
*Task: Object Detection*

* Learning Placeholders for Open-Set Recognition (CVPR 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhou_Learning_Placeholders_for_Open-Set_Recognition_CVPR_2021_paper.pdf)]<br>
*Datasets: SVHN, CIFAR10, CIFAR+10, CIFAR+50, Tiny-ImageNet*<br>
*Task: Image Classification*

* Few-Shot Open-Set Recognition by Transformation Consistency (CVPR 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Jeong_Few-Shot_Open-Set_Recognition_by_Transformation_Consistency_CVPR_2021_paper.pdf)]<br>
*Datasets: miniImageNet, tieredImageNet*<br>
*Task: Image Classification*

* OpenMix: Reviving Known Knowledge for Discovering Novel Visual Categories in an Open World (CVPR 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhong_OpenMix_Reviving_Known_Knowledge_for_Discovering_Novel_Visual_Categories_in_CVPR_2021_paper.pdf)]<br>
*Datasets: CIFAR10, CIFAR 100, Imagenet*<br>
*Task: Image Classification*

<!-- #### ICLR
#### NeurIPS-->
#### ICCV
* OpenGAN: Open-Set Recognition via Open Data Generation (ICCV 2021 Best honorable) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Kong_OpenGAN_Open-Set_Recognition_via_Open_Data_Generation_ICCV_2021_paper.pdf)]
[[Code](https://github.com/aimerykong/OpenGAN)]<br>
*Datasets: MNIST, SVHN,CIFAR10, TinyImageNet, Cityscapes*<br>
*Task: Image Classification*

* NGC: A Unified Framework for Learning With Open-World Noisy Data (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_NGC_A_Unified_Framework_for_Learning_With_Open-World_Noisy_Data_ICCV_2021_paper.pdf)]<br>
*Datasets: CIFAR10, CIFAR 100, TinyImageNet, Places-365*<br>
*Task: Out-of-Distribution Detection*

* Conditional Variational Capsule Network for Open Set Recognition (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Guo_Conditional_Variational_Capsule_Network_for_Open_Set_Recognition_ICCV_2021_paper.pdf)]
[[Code](https://github.com/guglielmocamporese/cvaecaposr)]<br>
*Datasets: MNIST, SVHN, CIFAR10,  CIFAR+10, CIFAR+50 and TinyImageNet*<br>
*Task: Image Classification*

* Deep Metric Learning for Open World Semantic Segmentation (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Cen_Deep_Metric_Learning_for_Open_World_Semantic_Segmentation_ICCV_2021_paper.pdf)]<br>
*Datasets: StreetHazards, Lost and  Found  and  Road  Anomaly*<br>
*Task: Semantic Segmentation*

* Towards Discovery and Attribution of Open-World GAN Generated Images (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Girish_Towards_Discovery_and_Attribution_of_Open-World_GAN_Generated_Images_ICCV_2021_paper.pdf)]
[[Code](https://github.com/Sharath-girish/openworld-gan)]<br>
*Datasets: CelebA, CelebA-HQ, ImageNet, LSUN Bedroom*<br>
*Task: Image Generation*

* Prototypical Matching and Open Set Rejection for Zero-Shot Semantic Segmentation (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Prototypical_Matching_and_Open_Set_Rejection_for_Zero-Shot_Semantic_Segmentation_ICCV_2021_paper.pdf)]<br>
*Datasets: Pascal VOC 2012, Pascal Context*<br>
*Task: Semantic Segmentation*

* Energy-Based Open-World Uncertainty Modeling for Confidence Calibration (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Energy-Based_Open-World_Uncertainty_Modeling_for_Confidence_Calibration_ICCV_2021_paper.pdf)]<br>
*Datasets: MNIST,CIFAR-10/100 and Tiny-ImageNet*<br>
*Task: Confidence Calibration*

* Trash To Treasure: Harvesting OOD Data With Cross-Modal Matching for Open-Set Semi-Supervised Learning (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Huang_Trash_To_Treasure_Harvesting_OOD_Data_With_Cross-Modal_Matching_for_ICCV_2021_paper.pdf)]<br>
*Datasets: CIFAR-10, Animal-10, Tiny-ImageNet, CIFAR100*<br>
*Task: Out-of-Distribution Detection*

* Towards Novel Target Discovery Through Open-Set Domain Adaptation (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Jing_Towards_Novel_Target_Discovery_Through_Open-Set_Domain_Adaptation_ICCV_2021_paper.pdf)]
[[Code](https://github.com/scottjingtt/SROSDA.git)]<br>
*Datasets: D2AwA; IAwA2*<br>
*Task: Image Classification*

#### ICML
* Learning bounds for open-set learning (ICML 2021) 
[[Paper](http://proceedings.mlr.press/v139/fang21c/fang21c.pdf)]
[[Code](https://github.com/Anjin-Liu/Openset_Learning_AOSR)]<br>
*Datasets: n MNIST, SVHN, CIFAR-10, CIFAR+10, CIFAR+50*<br>
*Task: Image Classification*

<!--#### IEEE-Access
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
### Older Papers
* Toward Open Set Recognition (TPAMI 2012) 
[[Paper](https://ieeexplore.ieee.org/document/6365193)]<br>
*Datasets: Caltech 256, ImageNet*<br>
*Task: Image Classification*

* Multi-class Open Set Recognition Using Probability of Inclusion (ECCV 2014) 
[[Paper](https://link.springer.com/chapter/10.1007/978-3-319-10578-9_26)]<br>
*Datasets:LETTER, MNIST*<br>
*Task: Image Classification*

* Towards Open World Recognition (CVPR 2015) 
[[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Bendale_Towards_Open_World_2015_CVPR_paper.pdf)]
[[Code](http://vast.uccs.edu/OpenWorld)]<br>
*Datasets: ImageNet 2010*<br>
*Task: Image Classification*

* Towards Open Set Deep Networks (OpenMax) (CVPR 2016) 
[[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Bendale_Towards_Open_Set_CVPR_2016_paper.pdf)]<br>
*Datasets: ILSVRC 2012 *<br>
*Task: Image Classification*

* Generative OpenMax for multi-class open set classification (BMVC 2017) 
[[Paper](http://www.bmva.org/bmvc/2017/papers/paper042/paper042.pdf)]<br>
*Datasets: : MNIST, HASYv2*<br>
*Task: Image Classification*

* Open-world Learning and Application to Product Classification (WWW 2019) 
[[Paper](https://arxiv.org/pdf/1809.06004v2.pdf)]
[[Code](https://www.cs.uic.edu/~hxu/)]<br>
*Datasets:  product descriptions from the Amazon Datasets*<br>
*Task: Image Classification*

* C2AE: Class Conditioned Auto-Encoder for Open-set Recognition (CVPR 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Oza_C2AE_Class_Conditioned_Auto-Encoder_for_Open-Set_Recognition_CVPR_2019_paper.pdf)]
[[Code](https://github.com/otkupjnoz/c2ae)]<br>
*Datasets: MNIST, SVHN, CIFAR10, CIFAR+10, CIFAR+50, TinyImageNet*<br>
*Task: Image Classification*

<!----------------------------------------------------------------------------------------------------------------------------------------------->
## Novel Class Discovery
<!----------------------------------------------------------------------------------------------------------------------------------------------->

### 2023 Papers
<!-- #### CVPR-->
#### ICLR
* Effective Cross-instance Positive Relations for Generalized Category Discovery (ICLR 2023 Submission) 
[[Paper](https://openreview.net/forum?id=hag85Gdq_RA)]<br>
*Datasets: CIFAR-10, CIFAR-100, and ImageNet-100, CUB-200, SCars, Herbarium19*<br>
*Task: Image Classification*

<!--#### NeurIPS
#### ICCV
#### ICML
#### IEEE-Access
#### ECCV
#### AAAI
#### TPAMI
#### CVPRw-->
#### WACV
* Scaling Novel Object Detection With Weakly Supervised Detection Transformers (WACV 2023) 
[[Paper](http://arxiv.org/abs/2207.05205)]<br>
*Datasets: Few-Shot  Object  Detec-tion (FSOD), FGVC-Aircraft, iNaturalist  2017, PASCAL VOC 2007*<br>
*Task: Object Detection*
<!--#### IJCV
#### BMVC
#### ICCw-->
#### Arxiv & Others
* Zero-Knowledge Zero-Shot Learning for Novel Visual Category Discovery (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2302.04427v1)]<br>
*Datasets:  Attribute Pascal and Yahoo (APY), Animals with Attributes2 (AWA2), Caltech-UCSD-Birds 200-2011 (CUB), SUN*<br>
*Task: Image Classification*
<!----------------------------------------------------------------------------------------------------------------------------------------------->
### 2022 Papers
#### CVPR
* Divide and Conquer: Compositional Experts for Generalized Novel Class Discovery (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Divide_and_Conquer_Compositional_Experts_for_Generalized_Novel_Class_Discovery_CVPR_2022_paper.pdf)]
[[Code](https://github.com/muliyangm/ComEx)]<br>
*Datasets: CIFAR-10; CIFAR100-50; CIFAR100-20; ImageNet*<br>
*Task: Image Classification*

* Generalized Category Discovery (CVPR 2022) 
[[Paper](https://arxiv.org/abs/2201.02609v2)]
[[Code](https://github.com/sgvaze/generalized-category-discovery)]<br>
*Datasets: CIFAR10,  CIFAR100, ImageNet-100*<br>
*Task: Image Classification*

* Novel Class Discovery in Semantic Segmentation (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhao_Novel_Class_Discovery_in_Semantic_Segmentation_CVPR_2022_paper.pdf)]
[[Code](https://ncdss.github.io)]<br>
*Datasets: PASCAL-5i dataset; the COCO-20i dataset*<br>
*Task: Semantic Segmentation*

#### ICLR
* Meta Discovery: Learning to Discover Novel Classes given Very Limited Data (ICLR 2022 spotlight) 
[[Paper](https://openreview.net/forum?id=MEpKGLsY8f)]
[[Code](https://github.com/Haoang97/MEDI)]<br>
*Datasets: CIFAR-10, CIFAR-100, SVHN, OmniGlot*<br>
*Task: Image Classification*

#### NeurIPS
* Learning to Discover and Detect Objects (NeurIPS 2022) 
[[Paper](https://arxiv.org/pdf/2210.10774.pdf)]
[[Code](https://vlfom.github.io/RNCDL)]<br>
*Datasets: COCOhalf+ LVIS; LVIS + Visual Genome*<br>
*Task: Object Detection*

* Grow and Merge: A Unified Framework for Continuous Categories Discovery (NeurIPS 2022) 
[[Paper](https://arxiv.org/abs/2210.04174v1)]<br>
*Datasets:  CIFAR-100, CUB-200, ImageNet-100, Stanford-Cars, FGVC-Aircraft, ImageNet-200*<br>
*Task: Image Classification*
<!-- #### ICCV
#### ICML
#### IEEE-Access-->
#### ECCV
* Generative Meta-Adversarial Network for Unseen Object Navigation (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136990295.pdf)]
[[Code](https://github.com/sx-zhang/GMAN.git)]<br>
*Datasets: AI2THOR  and RoboTHOR*<br>
*Task: Object Navigation*

* incDFM: Incremental Deep Feature Modeling for Continual Novelty Detection (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850581.pdf)]<br>
*Datasets: 1.CIFAR-10 (10 classes), 2. CIFAR-100 (super-classlevel, 20 classes), 3. EMNIST (26 classes) and 4. iNaturalist21 (phylumlevel, 9 classes)*<br>
*Task: Image Classification*

#### AAAI
* Self-Labeling Framework for Novel Category Discovery over Domains (AAAI 2022) 
[[Paper](https://ojs.aaai.org/index.php/AAAI/article/download/20224/version/18521/19983)]<br>
*Datasets: Office (Amazon, DSLR, Webcam); OfficeHome (art, clipart, product, and real); VisDA (synthetic and real)*<br>
*Task: Image Classification*

<!-- #### TPAMI -->
#### CVPRw
* Spacing Loss for Discovering Novel Categories (CVPRw 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022W/CLVision/papers/Joseph_Spacing_Loss_for_Discovering_Novel_Categories_CVPRW_2022_paper.pdf)]
[[Code](https://github.com/JosephKJ/Awesome-Novel-Class-Discovery)]<br>
*Datasets: CIFAR-10 and CIFAR-100*<br>
*Task: Image Classification*

#### WACV
* One-Class Learned Encoder-Decoder Network With Adversarial Context Masking for Novelty Detection (WACV 2022) 
[[Paper](https://openaccess.thecvf.com/content/WACV2022/papers/Jewell_One-Class_Learned_Encoder-Decoder_Network_With_Adversarial_Context_Masking_for_Novelty_WACV_2022_paper.pdf)]
[[Code](https://github.com/jewelltaylor/OLED)]<br>
*Datasets: MNIST, CIFAR-10, UCSD*<br>
*Task: Novelty Detection, Anomaly*

* COCOA: Context-Conditional Adaptation for Recognizing Unseen Classes in Unseen Domains (WACV 2022) 
[[Paper](https://openaccess.thecvf.com/content/WACV2022/papers/Mangla_COCOA_Context-Conditional_Adaptation_for_Recognizing_Unseen_Classes_in_Unseen_Domains_WACV_2022_paper.pdf)]<br>
*Datasets: DomainNet, DomainNet-LS*<br>
*Task: Domain Generalization and Novel Class Discovery*


<!-- #### IJCV-->
#### BMVC
* XCon: Learning with Experts for Fine-grained Category Discovery (BMVC 2022) 
[[Paper](https://arxiv.org/abs/2208.01898v1)]
[[Code](https://github.com/YiXXin/XCon)]<br>
*Datasets: CIFAR-10/100, ImageNet-100, CUB-200, Standford Cars, FGVC-Aircraft, and Oxford-IIIT Pet*<br>
*Task: Image Classification*

<!--#### ICCw-->
#### Arxiv & Others
* Mutual Information-guided Knowledge Transfer for Novel Class Discovery (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2206.12063v2)]


* Automatically Discovering Novel Visual Categories with Adaptive Prototype Learning (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2206.12063v2)]
[[Code](https://github.com/dvlab-research/Entity)]<br>
*Datasets: CIFAR10, CIFAR100, OmniGlot, ImageNet*<br>
*Task: Image Classification*

* Automatically Discovering Novel Visual Categories with Self-supervised Prototype Learning (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2208.00979v1)]
[[Code](https://github.com/dvlab-research/Entity)]<br>
*Datasets: CIFAR10, CIFAR100, OmniGlot, ImageNet*<br>
*Task: Image Classification*

* A Method for Discovering Novel Classes in Tabular Data (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2209.01217v3)]<br>
*Datasets: ForestCover Type, Letter Recognition, Human Activity Recognition, Satimage, Pen-Based Handwritten Digits Recognition, 1990 US Census Data, MNIST*<br>
*Task: Tabular Data Classification*

* A Closer Look at Novel Class Discovery from the Labeled Set (NeurIPSw 2022) 
[[Paper](https://arxiv.org/abs/2209.09120v4)]<br>
*Datasets: CIFAR100-50, ImageNet*<br>
*Task: Image Classification*

* Modeling Inter-Class and Intra-Class Constraints in Novel Class Discovery (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2210.03591v2)]<br>
*Datasets: CIFAR10, CIFAR100, ImageNet*<br>
*Task: Image Classification*

* A Simple Parametric Classification Baseline for Generalized Category Discovery (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2211.11727v1)]
[[Code](https://github.com/CVMI-Lab/SimGCD)]<br>
*Datasets: CIFAR10/100, ImageNet-100, Semantic Shift Benchmark (SSB, including CUB, Stanford Cars, and FGVC-Aircraft), Herbarium 19*<br>
*Task: Image Classification*

* Residual Tuning: Toward Novel Category Discovery Without Labels (TNNLS 2022) 
[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9690577&tag=1)]
[[Code](https://github.com/liuyudut/ResTune)]<br>
*Datasets: CIFAR-10, CIFAR-100, TinyImageNet*<br>
*Task: Image Classification*

* Open Set Domain Adaptation By Novel Class Discovery (ICME 2022) 
[[Paper](https://arxiv.org/abs/2203.03329v1)]<br>
*Datasets: Office-31, Office-Home, DomainNet*<br>
*Task: Image Classification and Domain Adaptation*

* Fine-grained Category Discovery under Coarse-grained supervision with Hierarchical Weighted Self-contrastive Learning (EMNLP 2022) 
[[Paper](https://arxiv.org/abs/2210.07733v1)]
[[Code](https://github.com/Lackel/Hierarchical_Weighted_SCL)]<br>
*Datasets: CLINC, Web of Science (WOS), HWU64*<br>
*Task: Text Classification*

<!----------------------------------------------------------------------------------------------------------------------------------------------->
### 2021 Papers
#### CVPR
* OpenMix: Reviving Known Knowledge for Discovering Novel Visual Categories in an Open World (CVPR 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhong_OpenMix_Reviving_Known_Knowledge_for_Discovering_Novel_Visual_Categories_in_CVPR_2021_paper.pdf)]<br>
*Datasets: CIFAR10, CIFAR 100, Imagenet*<br>
*Task: Object Detection*
<!--#### ICLR-->
#### NeurIPS
* Novel Visual Category Discovery with Dual Ranking Statistics and Mutual Knowledge Distillation (NeurIPS 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Jing_Towards_Novel_Target_Discovery_Through_Open-Set_Domain_Adaptation_ICCV_2021_paper.pdf)]
[[Code](https://github.com/DTennant/dual-rank-ncd)]<br>
*Datasets: CIFAR-10, CIFAR-100, ImageNet-1K, ImageNet-100, CUB-200, Stanford-Cars, FGVC aircraft*<br>
*Task: Image Classification*

#### ICCV
* Towards Novel Target Discovery Through Open-Set Domain Adaptation (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Jing_Towards_Novel_Target_Discovery_Through_Open-Set_Domain_Adaptation_ICCV_2021_paper.pdf)]
[[Code](https://github.com/scottjingtt/SROSDA.git)]<br>
*Datasets: D2AwA; IAwA2*<br>
*Task: Image Classification*

* A Unified Objective for Novel Class Discovery (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Fini_A_Unified_Objective_for_Novel_Class_Discovery_ICCV_2021_paper.pdf)]
[[Code](https://ncd-uno.github.ioc)]<br>
*Datasets: CIFAR10; CIFAR100-20; CIFAR100-50; ImageNet*<br>
*Task: Image Classification*

* The Surprising Impact of Mask-Head Architecture on Novel Class Segmentation (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Birodkar_The_Surprising_Impact_of_Mask-Head_Architecture_on_Novel_Class_Segmentation_ICCV_2021_paper.pdf)]
[[Code](https://git.io/deepmac)]<br>
*Datasets: VOC, COCO*<br>
*Task: Instance segmentation*

* The Pursuit of Knowledge: Discovering and Localizing Novel Categories Using Dual Memory (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Rambhatla_The_Pursuit_of_Knowledge_Discovering_and_Localizing_Novel_Categories_Using_ICCV_2021_paper.pdf)]<br>
*Datasets: PascalVOC2007-> COCO2014*<br>
*Task: Object Detection and Localization*


<!--#### ICML
#### IEEE-Access
#### ECCV
#### AAAI-->
#### TPAMI

* AutoNovel: Automatically Discovering and Learning Novel Visual Categories (TPAMI 2021) 
[[Paper](https://arxiv.org/abs/2106.15252v1)]
[[Code](http://www.robots.ox.ac.uk/~vgg/research/auto_novel/)]<br>
*Datasets: CIFAR10, CIFAR100, SVHN, OmniGlot, ImageNet*<br>
*Task: Image Classification*

<!--#### CVPRw
#### WACV
#### IJCV
#### BMVC
#### ICCw-->
#### Arxiv & Others

* End-to-end novel visual categories learning via auxiliary self-supervision (Neural Networks 2021) 
[[Paper](https://www.sciencedirect.com/science/article/pii/S0893608021000575)]<br>
*Datasets: CIFAR10, CIFAR100, SVHN*<br>
*Task: Image Classification*

* Progressive Self-Supervised Clustering With Novel Category Discovery (TCYB 2021) 
[[Paper](https://ieeexplore.ieee.org/document/9409777)]<br>
*Datasets: Coil20, Yeast, MSRA25, PalmData25, Abalone, USPS, Letter, MNIST*<br>
*Task: Self-Supervised Clustering*
<!----------------------------------------------------------------------------------------------------------------------------------------------->
### Older Papers

* Learning to Discover Novel Visual Categories via Deep Transfer Clustering (ICCV 2019) 
[[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Han_Learning_to_Discover_Novel_Visual_Categories_via_Deep_Transfer_Clustering_ICCV_2019_paper.pdf)][[Code](https://github.com/k-han/DTC)]<br>
*Datasets: ImageNet, OmniGlot, CIFAR-100, CIFAR-10, and SVHN*<br>
*Task: Image Classification*

* Open-World Class Discovery with Kernel Networks (ICDM 2020) 
[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9338431)]
[[Code](https://github.com/neu-spiral/OpenWorldKNet)]<br>
*Datasets: MNIST; Fashion-MNIST; CIFAR-100; RF-50*<br>
*Task: Image Classification*

* Automatically Discovering and Learning New Visual Categories with Ranking Statistics (ICLR 2020) 
[[Paper](https://arxiv.org/pdf/2002.05714v1.pdf)]
[[Code](http://www.robots.ox.ac.uk/~vgg/research/auto_novel)]<br>
*Datasets: CIFAR-10; CIFAR-100; SVHN*<br>
*Task: Image Classification*

* Multi-class classification without multi-class labels (ICLR 2019) 
[[Paper](https://openreview.net/forum?id=SJzR2iRcK7)]
[[Code](https://github.com/GT-RIPL/L2C)]<br>
*Datasets: CIFAR-10, CIFAR-100, ImageNet, MNIST*<br>
*Task: Image Classification*

* Learning to cluster in order to transfer across domains and tasks (ICLR 2018) 
[[Paper](https://openreview.net/pdf?id=ByRWCqvT-)]
[[Code](https://github.com/GT-RIPL/L2C)]<br>
*Datasets: Omniglot, Office-31, SVHN-TO-MNIST*<br>
*Task: Cross-domain Transfer Learning, Clustering*

* Neural network-based clustering using pairwise constraints (ICLRw 2016) 
[[Paper](https://arxiv.org/abs/1511.06321v5)]
[[Code](https://github.com/GT-RIPL/L2C)]<br>
*Datasets: MNIST, CIFAR10*<br>
*Task: Image Classification, Clustering*
<!----------------------------------------------------------------------------------------------------------------------------------------------->
## Open Vocabulary
<!----------------------------------------------------------------------------------------------------------------------------------------------->

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
<!----------------------------------------------------------------------------------------------------------------------------------------------->
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

* NOC-REK: Novel Object Captioning With Retrieved Vocabulary From External Knowledge (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Vo_NOC-REK_Novel_Object_Captioning_With_Retrieved_Vocabulary_From_External_Knowledge_CVPR_2022_paper.pdf)]<br>
*Datasets: COCO, Nocaps*<br>
*Task: Novel Object Captioning*

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
[[Paper](https://arxiv.org/pdf/2112.12143v2.pdf)]<br>
*Datasets: COCO, Localized Narrative (Loc. Narr.) test: PASCAL Context, PASCAL  VOC, ADE20k*<br>
*Task: Instance segmentation*

* Towards Open-vocabulary Scene Graph Generation with Prompt-based Finetuning (ECCV 2022) 
[[Paper](https://arxiv.org/pdf/2208.08165v3.pdf)]<br>
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
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890725.pdf)]
[[Code](https://github.com/MendelXu/zsseg.baseline)]<br>
*Datasets: COCO Stuff; Pascal VOC 2012; Cityscapes; Pascal Context; ADE20K*<br>
*Task: Semantic Segmentation*

* A Dataset for Interactive Vision-Language Navigation with Unknown Command Feasibility (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680304.pdf)]
[[Code](https://github.com/aburns4/MoTIF)]<br>
*Datasets: MoTIF*<br>
*Task: Vision-Language Navigation (Apps)*

* Acknowledging the Unknown for Multi-label Learning with Single Positive Labels (ECCV 2022) 
[[Paper](https://arxiv.org/abs/2203.16219v2)]
[[Code](https://github.com/Correr-Zhou/SPML-AckTheUnknown)]<br>
*Datasets: PASCAL VOC 2012 (VOC), MS-COCO 2014 (COCO), NUS-WIDE (NUS), and CUB-200-2011 (CUB)*<br>
*Task: Single Positive Multi-label Learning*

#### AAAI
* OVIS: Open-Vocabulary Visual Instance Search via Visual-Semantic Aligned Representation Learning (AAAI 2022) 
[[Paper](https://ojs.aaai.org/index.php/AAAI/article/download/20070/version/18367/19829)]<br>
*Datasets: OVIS40; OVIS1600*<br>
*Task: Visual Instance Search*

<!--#### TPAMI
#### CVPRw
#### WACV
#### IJCV-->
#### BMVC
* Partially-Supervised Novel Object Captioning Using Context from Paired Data (BMVC 2022) 
[[Paper](https://arxiv.org/abs/2109.05115v2)]<br>
*Datasets: MS COCO*<br>
*Task: Object Captioning*

* Open-vocabulary Semantic Segmentation with Frozen Vision-Language Models (BMVC 2022) 
[[Paper](https://arxiv.org/pdf/2210.15138v1.pdf)]
[[Code](https://yyh-rain-song.github.io/Fusioner_webpage/)]<br>
*Datasets: : PASCAL-5i, COCO-20i, FSS-1000, Mosaic-4*<br>
*Task: Semantic Segmentation*
<!-- #### ICCw -->

#### Arxiv & Others
* Describing Sets of Images with Textual-PCA (EMNLP 2022) 
[[Paper](https://arxiv.org/pdf/2210.12112v1.pdf)]
[[Code](https://github.com/OdedH/textual-pca)]<br>
*Datasets: CelebA; Stanford Cars; COCO-Horses; LSUN-Church*<br>
*Task: Text Generation for Sets of Images*
<!----------------------------------------------------------------------------------------------------------------------------------------------->
### 2021 Papers
#### CVPR
* Open-Vocabulary Object Detection Using Captions	(CVPR 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zareian_Open-Vocabulary_Object_Detection_Using_Captions_CVPR_2021_paper.pdf)]
[[Code](https://github.com/alirezazareian/ovr-cnn)]<br>
*Datasets: COCO Objects, COCO Captions*<br>
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
## Fine Grained
<!----------------------------------------------------------------------------------------------------------------------------------------------->

### 2023 Papers
<!-- #### CVPR -->
#### ICLR
* This Looks Like It Rather Than That: ProtoKNN For Similarity-Based Classifiers (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=lh-HRYxuoRr)]<br>
*Datasets: CUB200-2011, Stanford Dogs, Stanford Cars*<br>
*Task: Image Classification*

<!-- #### NeurIPS
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
### 2022 Papers
#### CVPR

* Dual Cross-Attention Learning for Fine-Grained Visual Categorization and Object Re-Identification (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhu_Dual_Cross-Attention_Learning_for_Fine-Grained_Visual_Categorization_and_Object_Re-Identification_CVPR_2022_paper.pdf)]<br>
*Datasets: CUB-200-2011, Stanford  Cars, FGVC-Aircraft ; For Re-ID, we use four standard benchmarks:  Mar-ket1501,  DukeMTMC-ReID,  MSMT17 for Person Re-ID and VeRi-776 for Vehicle Re-ID*<br>
*Task: Image Classification*

* Knowledge Mining With Scene Text for Fine-Grained Recognition (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Knowledge_Mining_With_Scene_Text_for_Fine-Grained_Recognition_CVPR_2022_paper.pdf)]
[[Code](https://github.com/lanfeng4659/KnowledgeMiningWithSceneText)]<br>
*Datasets: Con-Text; Drink Bottle; Crowd Activity*<br>
*Task: Image Classification*

* Few-Shot Font Generation by Learning Fine-Grained Local Styles (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Tang_Few-Shot_Font_Generation_by_Learning_Fine-Grained_Local_Styles_CVPR_2022_paper.pdf)]<br>
*Datasets: UFUC and UFSC*<br>
*Task: Image Generation*

* Estimating Fine-Grained Noise Model via Contrastive Learning (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zou_Estimating_Fine-Grained_Noise_Model_via_Contrastive_Learning_CVPR_2022_paper.pdf)]<br>
*Datasets: SIDD,  CRVD and PMRID*<br>
*Task: Image Denoising*

* Fine-Grained Object Classification via Self-Supervised Pose Alignment (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Fine-Grained_Object_Classification_via_Self-Supervised_Pose_Alignment_CVPR_2022_paper.pdf)]
[[Code](https://github.com/yangxh11/P2P-Net)]<br>
*Datasets: Caltech-UCSD Birds (CUB), Stanford Cars (CAR), FGVC Aircraft (AIR)*<br>
*Task: Image Classification*

* Fine-Grained Predicates Learning for Scene Graph Generation (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Lyu_Fine-Grained_Predicates_Learning_for_Scene_Graph_Generation_CVPR_2022_paper.pdf)]
[[Code](https://github.com/XinyuLyu/FGPL)]<br>
*Datasets: Visual Genome*<br>
*Task: Scene Graph Generation*

* Dynamic MLP for Fine-Grained Image Classification by Leveraging Geographical and Temporal Information (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Dynamic_MLP_for_Fine-Grained_Image_Classification_by_Leveraging_Geographical_and_CVPR_2022_paper.pdf)]
[[Code](https://github.com/megvii-research/DynamicMLPForFinegrained)]<br>
*Datasets: iNaturalist 2017, 2018, 2021 and YFCC100M-GEO100*<br>
*Task: Image Classification*

* Task Discrepancy Maximization for Fine-Grained Few-Shot Classification (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_Task_Discrepancy_Maximization_for_Fine-Grained_Few-Shot_Classification_CVPR_2022_paper.pdf)]<br>
*Datasets: CUB-200-2011, Aircraft, meta-iNat, tiered meta-iNat,Stanford-Cars, Stanford-Dogs, and Oxford-Pets*<br>
*Task: Image Classification*

* FaceVerse: A Fine-Grained and Detail-Controllable 3D Face Morphable Model From a Hybrid Dataset (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_FaceVerse_A_Fine-Grained_and_Detail-Controllable_3D_Face_Morphable_Model_From_CVPR_2022_paper.pdf)]
[[Code](https://github.com/emilianavt/OpenSeeFace)]<br>
*Datasets: FFHQ*<br>
*Task: 3D Face Morphable Model*

* SphericGAN: Semi-Supervised Hyper-Spherical Generative Adversarial Networks for Fine-Grained Image Synthesis (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_SphericGAN_Semi-Supervised_Hyper-Spherical_Generative_Adversarial_Networks_for_Fine-Grained_Image_Synthesis_CVPR_2022_paper.pdf)]<br>
*Datasets: CUB-200/FaceScrub-100/Stanford-Cars*<br>
*Task: Image Generation*

* Attentive Fine-Grained Structured Sparsity for Image Restoration (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Oh_Attentive_Fine-Grained_Structured_Sparsity_for_Image_Restoration_CVPR_2022_paper.pdf)]
[[Code](https://github.com/JungHunOh/SLS_CVPR2022)]<br>
*Datasets: GOPRO; DIV2K; Set14, B100  and Urban100*<br>
*Task: Image super-resolution*

* PatchNet: A Simple Face Anti-Spoofing Framework via Fine-Grained Patch Recognition (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_PatchNet_A_Simple_Face_Anti-Spoofing_Framework_via_Fine-Grained_Patch_Recognition_CVPR_2022_paper.pdf)]<br>
*Datasets: OULU-NPU (denoted asO), SiW(denoted as S), CASIA-FASD (denotedas C), Replay-Attack (denoted as I), MSU-MFSD*<br>
*Task: Face anti-spoofing*

* GrainSpace: A Large-Scale Dataset for Fine-Grained and Domain-Adaptive Recognition of Cereal Grains (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Fan_GrainSpace_A_Large-Scale_Dataset_for_Fine-Grained_and_Domain-Adaptive_Recognition_of_CVPR_2022_paper.pdf)]
[[Code](https://github.com/hellodfan/GrainSpace)]<br>
*Datasets: GrainSpace*<br>
*Task: Grain Appearance Inspection (GAI)*

<!-- #### ICLR
#### NeurIPS
#### ICCV
#### ICML
#### IEEE-Access-->
#### ECCV
* Zero-Shot Attribute Attacks on Fine-Grained Recognition Models (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650257.pdf)]<br>
*Datasets: Caltech-UCSD Birds-200-2011(CUB), Animal with Attributes (AWA2) and SUN Attribute (SUN)*<br>
*Task: Image Classification*

* MvDeCor: Multi-View Dense Correspondence Learning for Fine-Grained 3D Segmentation (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620538.pdf)]
[[Code](https://nv-tlabs.github.io/MvDeCor)]<br>
*Datasets: 3D Segmentation*<br>
*Task: PartNet; Render People; ShapeNet-Part*

* Fine-Grained Data Distribution Alignment for Post-Training Quantization (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710070.pdf)]
[[Code](https://github.com/zysxmu/FDDA)]<br>
*Datasets: ResNet-18, MobileNetV1, MobileNetV2 and RegNet-600MF*<br>
*Task: Network Quantization*

* Deep Ensemble Learning by Diverse Knowledge Distillation for Fine-Grained Object Classification (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710501.pdf)]<br>
*Datasets: Stanford Dogs, Stanford Cars, CUB-200-2011, CIFAR-10, and CIFAR-100*<br>
*Task: Image Classification*

* RDO-Q: Extremely Fine-Grained Channel-Wise Quantization via Rate-Distortion Optimization (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720156.pdf)]<br>
*Datasets: ResNet-18, ResNet-34, ResNet-50 andMobileNet-v2, on the ImageNet dataset*<br>
*Task: Network Quantization*

* Rethinking Robust Representation Learning under Fine-Grained Noisy Faces (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720605.pdf)]<br>
*Datasets: MS1MV0; MS1MV3*<br>
*Task: Face Recognition*

* SEMICON: A Learning-to-Hash Solution for Large-Scale Fine-Grained Image Retrieval (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740518.pdf)]
[[Code](https://github.com/NJUST-VIPGroup/SEMICON)]<br>
*Datasets: CUB200-2011, Aircraft and Food101, NABirds, VegFru*<br>
*Task: Image Retrieval*

* Where to Focus: Investigating Hierarchical Attention Relationship for Fine-Grained Visual Classification (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840056.pdf)]
[[Code](https://github.com/visiondom/CHRF)]<br>
*Datasets: CUB; Butterfly-200; VegFru; FGVC-Aircraft; Stanford  Cars*<br>
*Task: Image Classification*

* Improving Fine-Grained Visual Recognition in Low Data Regimes via Self-Boosting Attention Mechanism (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850444.pdf)]
[[Code](https://github.com/GANPerf/SAM)]<br>
*Datasets: Caltech-UCSD Birds (CUB-200-2011), Stanford Cars and FGVC-Aircraft*<br>
*Task: Image Classification*

* Conditional Stroke Recovery for Fine-Grained Sketch-Based Image Retrieval (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860708.pdf)]
[[Code](https://github.com/1069066484/CSR-ECCV2022)]<br>
*Datasets: Sketchy, QMUL-Shoe, QMUL-Chair, and QMUL-ShoeV2*<br>
*Task: Image Retrieval*

* Fine-Grained Fashion Representation Learning by Online Deep Clustering (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870019.pdf)]<br>
*Datasets: FashionAI; DARN*<br>
*Task: Image Classification*

* Hierarchical Memory Learning for Fine-Grained Scene Graph Generation (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870263.pdf)]
[[Code]]<br>
*Datasets: Visual Genome (VG)*<br>
*Task: Scene Graph Detection*

* Fine-Grained Scene Graph Generation with Data Transfer (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870402.pdf)]
[[Code](https://github.com/waxnkw/IETrans-SGG.pytorch)]<br>
*Datasets: VG-50*<br>
*Task: Scene Graph Detection*

* TransFGU: A Top-down Approach to Fine-Grained Unsupervised Semantic Segmentation (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890072.pdf)]
[[Code](https://github.com/damo-cv/TransFGU)]<br>
*Datasets: COCO-Stuff; Cityscapes; Pascal-VOC; LIP*<br>
*Task: Semantic Segmentation*

* Fine-Grained Egocentric Hand-Object Segmentation: Dataset, Model, and Applications (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890125.pdf)]
[[Code](https://github.com/owenzlz/EgoHOS)]<br>
*Datasets: EPIC-KITCHENS; Ego4d; THU-READ; Escape Room*<br>
*Task: Semantic Segmentation*

* Word-Level Fine-Grained Story Visualization (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136960342.pdf)]
[[Code](https://github.com/mrlibw/Word-Level-Story-Visualization)]<br>
*Datasets: Pororo-SV; CLEVR-SV*<br>
*Task: Story Visualization*

* Fine-Grained Visual Entailment (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136960393.pdf)]
[[Code](https://github.com/SkrighYZ/FGVE)]<br>
*Datasets: VE+AMR→KE; VE→KE*<br>
*Task: Visual Entailment*

* Adaptive Fine-Grained Sketch-Based Image Retrieval (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136970160.pdf)]<br>
*Datasets: Sketchy (Category Level); Shoe-V2 (User Level)*<br>
*Task: Image Retrieval*


<!--#### AAAI
#### TPAMI
#### CVPRw
#### WACV
#### IJCV-->
#### BMVC

* XCon: Learning with Experts for Fine-grained Category Discovery (BMVC 2022) 
[[Paper](https://arxiv.org/abs/2208.01898v1)]
[[Code](https://github.com/YiXXin/XCon)]<br>
*Datasets: CIFAR-10/100, ImageNet-100, CUB-200, Standford Cars, FGVC-Aircraft, and Oxford-IIIT Pet*<br>
*Task: Image Classification*
<!--#### ICCw-->

#### Arxiv & Others

* Fine-grained Category Discovery under Coarse-grained supervision with Hierarchical Weighted Self-contrastive Learning (EMNLP 2022) 
[[Paper](https://arxiv.org/abs/2210.07733v1)]
[[Code](https://github.com/Lackel/Hierarchical_Weighted_SCL)]<br>
*Datasets: CLINC, Web of Science (WOS), HWU64*<br>
*Task: Text Classification*

<!----------------------------------------------------------------------------------------------------------------------------------------------->
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
## Long Tail
<!----------------------------------------------------------------------------------------------------------------------------------------------->

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

<!----------------------------------------------------------------------------------------------------------------------------------------------->
### 2022 Papers
#### CVPR
* C2AM Loss: Chasing a Better Decision Boundary for Long-Tail Object Detection (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_C2AM_Loss_Chasing_a_Better_Decision_Boundary_for_Long-Tail_Object_CVPR_2022_paper.pdf)]<br>
*Datasets: LVIS*<br>
*Task: Object Detection*

* Long-Tail Recognition via Compositional Knowledge Transfer (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Parisot_Long-Tail_Recognition_via_Compositional_Knowledge_Transfer_CVPR_2022_paper.pdf)]<br>
*Datasets: ImageNet-LT and Places-LT*<br>
*Task: Image Classification*

* Relieving Long-Tailed Instance Segmentation via Pairwise Class Balance (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Relieving_Long-Tailed_Instance_Segmentation_via_Pairwise_Class_Balance_CVPR_2022_paper.pdf)]<br>
*Datasets: LVIS*<br>
*Task: Instance Segmentation*

* Relieving Long-Tailed Instance Segmentation via Pairwise Class Balance (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Relieving_Long-Tailed_Instance_Segmentation_via_Pairwise_Class_Balance_CVPR_2022_paper.pdf)]<br>
*Datasets: LVIS*<br>
*Task: Instance Segmentation*

* RelTransformer: A Transformer-Based Long-Tail Visual Relationship Recognition (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_RelTransformer_A_Transformer-Based_Long-Tail_Visual_Relationship_Recognition_CVPR_2022_paper.pdf)]
[[Code](https://github.com/Vision-CAIR/ RelTransformer)]<br>
*Datasets: GQA-LT; VG8K-LT; VG200*<br>
*Task: Relationship Recognition*

* Long-Tailed Recognition via Weight Balancing (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Alshammari_Long-Tailed_Recognition_via_Weight_Balancing_CVPR_2022_paper.pdf)]
[[Code](https://github.com/ShadeAlsha/LTR-weight-balancing)]<br>
*Datasets: CIFAR100-LT; ImageNet-LT; iNaturalist*<br>
*Task: Image Classification*

* Equalized Focal Loss for Dense Long-Tailed Object Detection (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Equalized_Focal_Loss_for_Dense_Long-Tailed_Object_Detection_CVPR_2022_paper.pdf)]
[[Code](https://github.com/ModelTC/EOD)]<br>
*Datasets: LVIS v1*<br>
*Task: Object Detection*

* Targeted Supervised Contrastive Learning for Long-Tailed Recognition (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Targeted_Supervised_Contrastive_Learning_for_Long-Tailed_Recognition_CVPR_2022_paper.pdf)]<br>
*Datasets: CIFAR-10-LT  &  CIFAR-100-LT; ImageNet-LT; iNaturalist*<br>
*Task: Image Classification*

* Balanced Contrastive Learning for Long-Tailed Visual Recognition (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhu_Balanced_Contrastive_Learning_for_Long-Tailed_Visual_Recognition_CVPR_2022_paper.pdf)]<br>
*Datasets: CIFAR-10-LT  &  CIFAR-100-LT; ImageNet-LT; iNaturalist*<br>
*Task: Image Classification*

* Trustworthy Long-Tailed Classification (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Trustworthy_Long-Tailed_Classification_CVPR_2022_paper.pdf)]
[[Code]]<br>
*Datasets: (CIFAR-10-LT, CIFAR-100-LT and ImageNet-LT) and three balanced OOD datasets (SVHN, ImageNet-open and Places-open)*<br>
*Task: Image Classification*

* Nested Collaborative Learning for Long-Tailed Visual Recognition (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Nested_Collaborative_Learning_for_Long-Tailed_Visual_Recognition_CVPR_2022_paper.pdf)]
[[Code](https://github.com/Bazinga699/NCL)]<br>
*Datasets: CIFAR-10-LT  &  CIFAR-100-LT; ImageNet-LT; iNaturalist; Places-LT*<br>
*Task: Image Classification*

* The Majority Can Help the Minority: Context-Rich Minority Oversampling for Long-Tailed Classification (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Park_The_Majority_Can_Help_the_Minority_Context-Rich_Minority_Oversampling_for_CVPR_2022_paper.pdf)]
[[Code](https://github.com/naver-ai/cmo)]<br>
*Datasets: CIFAR-10-LT, CIFAR-100-LT, ImageNet-LT, and iNaturalist2018*<br>
*Task: Image Classification*

* Retrieval Augmented Classification for Long-Tail Visual Recognition (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Long_Retrieval_Augmented_Classification_for_Long-Tail_Visual_Recognition_CVPR_2022_paper.pdf)]<br>
*Datasets: iNaturalist; Places365-LT*<br>
*Task: Image Retrieval*

* Adaptive Hierarchical Representation Learning for Long-Tailed Object Detection (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Adaptive_Hierarchical_Representation_Learning_for_Long-Tailed_Object_Detection_CVPR_2022_paper.pdf)]<br>
*Datasets: LVIS*<br>
*Task: Object Detection*

* Long-Tailed Visual Recognition via Gaussian Clouded Logit Adjustment (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Long-Tailed_Visual_Recognition_via_Gaussian_Clouded_Logit_Adjustment_CVPR_2022_paper.pdf)]
[[Code](https://github.com/Keke921/GCLLoss)]<br>
*Datasets: CIFAR-10-LT  &  CIFAR-100-LT; ImageNet-LT; iNaturalist; Places-LT*<br>
*Task: Image Classification*

<!-- #### ICLR
#### NeurIPS
#### ICCV-->
#### ICML
* Open-Sampling: Exploring Out-of-Distribution Data for Re-balancing long-tailed Datasets (ICML 2022) 
[[Paper](https://proceedings.mlr.press/v162/wei22c/wei22c.pdf)]
[[Code](https://github.com/hongxin001/logitnorm_ood)]<br>
*Datasets: long-tailed CIFAR10/100, CelebA-5, Places-LT*<br>
*Task: Out-of-Distribution Detection, Image Classification*

<!-- #### IEEE-Access-->

#### ECCV
* Long-Tail Detection with Effective Class-Margins (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680684.pdf)]
[[Code](https://github.com/janghyuncho/ECM-Loss)]<br>
*Datasets: LVIS v1.0 and OpenImages*<br>
*Task: Image Classification*

* Improving the Intra-Class Long-Tail in 3D Detection via Rare Example Mining (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700155.pdf)]<br>
*Datasets: Waymo Open Dataset (camera+LiDAR)*<br>
*Task: Track Mining*

* Long-Tailed Instance Segmentation Using Gumbel Optimized Loss (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700349.pdf)]
[[Code](https://github.com/kostas1515/GOL)]<br>
*Datasets: LVISv1; LVISv0.5; CIFAR100-LT; ImageNet-LT, Places-LT*<br>
*Task: Image Classification*

* Learning with Free Object Segments for Long-Tailed Instance Segmentation (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700648.pdf)]
[[Code](https://github.com/czhang0528/FreeSeg)]<br>
*Datasets: LVISv1; COCO-LT*<br>
*Task: Instance segmentation*

* Improving GANs for Long-Tailed Data through Group Spectral Regularization (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750423.pdf)]
[[Code](https://sites.google.com/view/gsr-eccv22)]<br>
*Datasets: CIFAR-10 and LSUN*<br>
*Task: Image Classification*

* Constructing Balance from Imbalance for Long-Tailed Image Recognition (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800036.pdf)]
[[Code](https://github.com/silicx/DLSA)]<br>
*Datasets: ImageNet-LT, Places-LT, and iNaturalist18*<br>
*Task: Image Classification*

* On Multi-Domain Long-Tailed Recognition, Imbalanced Domain Generalization and Beyond (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800054.pdf)]
[[Code](https://github.com/YyzHarry/multi-domain-imbalance)]<br>
*Datasets: VLCS-MLT; PACS-MLT; OfficeHome-MLT; TerraInc-MLT; DomainNet-MLT*<br>
*Task: Image Classification*


* Tackling Long-Tailed Category Distribution under Domain Shifts (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830706.pdf)]
[[Code](https://xiaogu.site/LTDS)]<br>
*Datasets: AWA2-LTS; ImageNet-LTS*<br>
*Task: Image Classification*

* Towards Calibrated Hyper-Sphere Representation via Distribution Overlap Coefficient for Long-Tailed Learning (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840176.pdf)]
[[Code](https://github.com/VipaiLab/vMF_OP)]<br>
*Datasets: ADE20K; LVIS-v1.0*<br>
*Task: Image Classification*

* SAFA: Sample-Adaptive Feature Augmentation for Long-Tailed Image Classification (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840578.pdf)]<br>
*Datasets: CIFAR-LT-10,CIFAR-LT-100, Places-LT, ImageNet-LT, and iNaturalist2018*<br>
*Task: Image Classification*

* Breadcrumbs: Adversarial Class-Balanced Sampling for Long-Tailed Recognition (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840628.pdf)]
[[Code](https://github.com/BoLiu-SVCL/Breadcrumbs)]<br>
*Datasets: Places-LT, ImageNet-LT, and iNaturalist2018*<br>
*Task: Image Classification*

* Invariant Feature Learning for Generalized Long-Tailed Classification (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840698.pdf)]
[[Code](https://github.com/KaihuaTang/Generalized-Long-Tailed-Benchmarks.pytorch)]<br>
*Datasets: ImageNet-GLT; MSCOCO-GLT*<br>
*Task: Image Classification*

* VL-LTR: Learning Class-Wise Visual-Linguistic Representation for Long-Tailed Visual Recognition (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850072.pdf)]
[[Code](https://github.com/ChangyaoTian/VL-LTR)]<br>
*Datasets: ImageNet-LT, Places-LT, and iNaturalist 2018*<br>
*Task: Image Classification*

* Identifying Hard Noise in Long-Tailed Sample Distribution (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860725.pdf)]
[[Code](https://github.com/yxymessi/H2E-Framework)]<br>
*Datasets: ImageNet-NLT, Animal10-NLT and Food101-NLT; Red Mini-ImageNet, Animal-10N and Food-101N*<br>
*Task: Image Classification*

* Long-Tailed Class Incremental Learning (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930486.pdf)]
[[Code](https://github.com/xialeiliu/Long-Tailed-CIL)]<br>
*Datasets: CIFAR-100 and ImageNet-Subset with 100 classes*<br>
*Task: Image Classification*

<!--#### AAAI-->
#### TPAMI
* Open Long-Tailed RecognitionIn A Dynamic World	(TPAMI 2022) 
[[Paper](https://arxiv.org/pdf/2208.08349v1.pdf)]
[[Code](https://liuziwei7.github.io/projects/LongTail.html)]<br>
*Datasets: CIFAR-10-LT,CIFAR-100-LT, and iNaturalist-18, Places-LT,  MS1M-LT, SUN-LT*<br>
*Task: Image Classification*

#### CVPRw
* A Two-Stage Shake-Shake Network for Long-Tailed Recognition of SAR Aerial View Objects (CVPRw 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022W/PBVS/papers/Li_A_Two-Stage_Shake-Shake_Network_for_Long-Tailed_Recognition_of_SAR_Aerial_CVPRW_2022_paper.pdf)]
[[Code](https://codalab.lisn.upsaclay.fr/competitions/1388)]<br>
*Datasets: PBVS @ CVPR 2022 Multi-modal Aerial View Object Classification Challenge Track 1 (SAR images)*<br>
*Task: Image Classification*

<!--#### WACV
#### IJCV
#### BMVC
#### ICCw
#### Arxiv & Others-->

<!----------------------------------------------------------------------------------------------------------------------------------------------->
### 2021 Papers
<!--#### CVPR
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
## Anomaly Detection
<!----------------------------------------------------------------------------------------------------------------------------------------------->
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
<!----------------------------------------------------------------------------------------------------------------------------------------------->
### 2022 Papers
#### CVPR
* Learning Second Order Local Anomaly for General Face Forgery Detection (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Fei_Learning_Second_Order_Local_Anomaly_for_General_Face_Forgery_Detection_CVPR_2022_paper.pdf)]<br>
*Datasets: Face-Forensics++ (FF++), Celeb-DF v2 (CD2), Deep-fakeDetection Dataset (DFD), and FaceShifter (Fshi)*<br>
<!-- #### ICLR
#### NeurIPS
#### ICCV
#### ICML
#### IEEE-Access-->
#### ECCV
* Hierarchical Semi-Supervised Contrastive Learning for Contamination-Resistant Anomaly Detection (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850107.pdf)]
[[Code](https://github.com/GaoangW/HSCL)]<br>
*Datasets: CIFAR-10, CIFAR-100, ImageNet (FIX), SVHN, and LSUN (FIX)*<br>

* Locally Varying Distance Transform for Unsupervised Visual Anomaly Detection (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900353.pdf)]
[[Code](https://www.kind-of-works.com)]<br>
*Datasets: MNIST; STL-10; Internet STL-10; MIT-Places-5; CIFAR-10; CatVsDog; Fashion-MNIST*<br>

* Natural Synthetic Anomalies for Self-Supervised Anomaly Detection and Localization (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136910459.pdf)]
[[Code](https://github.com/hmsch/natural-synthetic-anomalies)]<br>
*Datasets: MVTecAD; rCXR*<br>

* DSR – A Dual Subspace Re-Projection Network for Surface Anomaly Detection (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136910526.pdf)]
[[Code](https://github.com/VitjanZ/DSR_anomaly_detection)]<br>
*Datasets: KSDD2; ImageNet*<br>

* Pixel-Wise Energy-Biased Abstention Learning for Anomaly Segmentation on Complex Urban Driving Scenes (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136990242.pdf)]
[[Code](https://github.com/tianyu0207/PEBAL)]<br>
*Datasets: LostAndFound; Fishyscapes; Road Anomaly*<br>
<!--#### AAAI
#### TPAMI-->
#### CVPRw
* Autoencoders - A Comparative Analysis in the Realm of Anomaly Detection (CVPRw 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022W/WiCV/papers/Schneider_Autoencoders_-_A_Comparative_Analysis_in_the_Realm_of_Anomaly_CVPRW_2022_paper.pdf)]<br>
*Datasets: CIFAR10, MNIST*<br>

#### WACV
* One-Class Learned Encoder-Decoder Network With Adversarial Context Masking for Novelty Detection (WACV 2022) 
[[Paper](https://openaccess.thecvf.com/content/WACV2022/papers/Jewell_One-Class_Learned_Encoder-Decoder_Network_With_Adversarial_Context_Masking_for_Novelty_WACV_2022_paper.pdf)]
[[Code](https://github.com/jewelltaylor/OLED)]<br>
*Datasets: MNIST, CIFAR-10, UCSD*<br>
*Task: Novelty Detection, Anomaly*
<!--#### IJCV
#### BMVC
#### ICCw
#### Arxiv & Others-->
<!----------------------------------------------------------------------------------------------------------------------------------------------->
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
# Video Open World Papers
<!----------------------------------------------------------------------------------------------------------------------------------------------->

<!----------------------------------------------------------------------------------------------------------------------------------------------->
## Zero-Shot Learning Videos
<!----------------------------------------------------------------------------------------------------------------------------------------------->

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
<!----------------------------------------------------------------------------------------------------------------------------------------------->
### 2022 Papers
#### CVPR
* Uni-Perceiver: Pre-Training Unified Architecture for Generic Perception for Zero-Shot and Few-Shot Tasks (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhu_Uni-Perceiver_Pre-Training_Unified_Architecture_for_Generic_Perception_for_Zero-Shot_and_CVPR_2022_paper.pdf)]<br>
*Datasets: ImageNet-21k;  Kinetics-700 and Moments in Time;  BookCorpora & English  Wikipedia  (Books&Wiki)  and  PAQ; COCO Caption, SBUCaptions  (SBU),  Visual  Genome,  CC3M, CC12M and YFCC; Flickr30k, MSVD,VQA ,and GLUE*<br>
*Task: Image-Text Retreival; Image and Video Classification*

* Cross-Modal Representation Learning for Zero-Shot Action Recognition (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Lin_Cross-Modal_Representation_Learning_for_Zero-Shot_Action_Recognition_CVPR_2022_paper.pdf)]
[[Code](https://github.com/microsoft/ResT)]<br>
*Datasets: Kinetics ->  UCF101, HMDB51, and ActivityNet*<br>
*Task: Action Recognition*

* Audio-Visual Generalised Zero-Shot Learning With Cross-Modal Attention and Language (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Mercea_Audio-Visual_Generalised_Zero-Shot_Learning_With_Cross-Modal_Attention_and_Language_CVPR_2022_paper.pdf)]
[[Code](https://github.com/ExplainableML/AVCA-GZSL)]<br>
*Datasets: VGGSound; UCF101; ActivityNet*<br>
*Task: Action Recognition*

* Alignment-Uniformity Aware Representation Learning for Zero-Shot Video Classification (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Pu_Alignment-Uniformity_Aware_Representation_Learning_for_Zero-Shot_Video_Classification_CVPR_2022_paper.pdf)]
[[Code](https://github.com/ShipuLoveMili/CVPR2022-AURL)]<br>
*Datasets: Kinetics-700 -> UCF101, HMDB51*<br>
*Task: Action Recognition*

<!-- #### ICLR
#### NeurIPS
#### ICCV
#### ICML
#### IEEE-Access-->
#### ECCV
* Temporal and cross-modal attention foraudio-visual zero-shot learning (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800474.pdf)]
[[Code](https://github.com/ExplainableML/TCAF-GZSL)]<br>
*Datasets: UCF-GZSL^cls, VGGSound-GZSL^cls, and ActivityNet-GZSL^cls1*<br>
*Task: Action Recognition*

* CLASTER: Clustering with Reinforcement Learning for Zero-Shot Action Recognition (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800177.pdf)]
[[Code](https://sites.google.com/view/claster-zsl/home)]<br>
*Datasets: Olympic Sports; UCF-101; HMDB-51*<br>
*Task: Action Recognition*

* Rethinking Zero-Shot Action Recognition: Learning from Latent Atomic Actions (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640102.pdf)]<br>
*Datasets: KineticsZSAR, HMDB51, and UCF101*<br>
*Task: Action Recognition*

* Zero-Shot Temporal Action Detection via Vision-Language Prompting (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630667.pdf)]
[[Code](https://github.com/sauradip/STALE)]<br>
*Datasets: THUMOS14; ActivityNet v1.3*<br>
*Task: Temporal Action Detection (TAD)*
<!--#### AAAI
#### TPAMI
#### CVPRw
#### WACV
#### IJCV
#### BMVC
#### ICCw
#### Arxiv & Others-->
<!----------------------------------------------------------------------------------------------------------------------------------------------->
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
## Out-of-Distribution Detection Videos
<!----------------------------------------------------------------------------------------------------------------------------------------------->

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
<!----------------------------------------------------------------------------------------------------------------------------------------------->
### 2022 Papers
#### CVPR
* Unknown-Aware Object Detection: Learning What You Don't Know From Videos in the Wild (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Du_Unknown-Aware_Object_Detection_Learning_What_You_Dont_Know_From_Videos_CVPR_2022_paper.pdf)]
[[Code](https://github.com/deeplearning-wisc/stud)]<br>
*Datasets: (Videos -> Images) BDD100K and Youtube-Video Instance Segmentation(Youtube-VIS)  2021 (ID) - MS-COCO and nuImages (OOD)*<br>
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
### Older Papers
* Uncertainty-aware audiovisual activity recognition using deep bayesian variational inference (ICCV 2019) 
[[Paper](https://arxiv.org/pdf/1811.10811v3.pdf)]<br>
*Datasets: MiT*<br>
*Task: Audiovisual Action Recognition*

* Bayesian activity recognition using variational inference (NeurIPS 2018) 
[[Paper](https://arxiv.org/pdf/1811.03305v2.pdf)]<br>
*Datasets:  MiT video activity recognition dataset*<br>
*Task: Action Recognition*

<!----------------------------------------------------------------------------------------------------------------------------------------------->
##  Open-set Recognition Videos
<!----------------------------------------------------------------------------------------------------------------------------------------------->

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
#### CVPRw-->
#### WACV
* Reconstructing Humpty Dumpty: Multi-Feature Graph Autoencoder for Open Set Action Recognition (WACV 2023) 
[[Paper](http://arxiv.org/abs/2212.06023)]
[[Code](https://github.com/Kitware/graphautoencoder)]<br>
*Datasets:  HMDB-51, UCF-101*<br>
*Task: Action Recognition*
<!--#### IJCV
#### BMVC
#### ICCw
#### Arxiv & Others-->
<!----------------------------------------------------------------------------------------------------------------------------------------------->
### 2022 Papers
#### CVPR
* Opening Up Open World Tracking (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Opening_Up_Open_World_Tracking_CVPR_2022_paper.pdf)]
[[Code](https://openworldtracking.github.io)]<br>
*Datasets: TAO-OW*<br>
*Task: Object Tracking*

* OpenTAL: Towards Open Set Temporal Action Localization (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Bao_OpenTAL_Towards_Open_Set_Temporal_Action_Localization_CVPR_2022_paper.pdf)]
[[Code](https://www.rit.edu/actionlab/opental)]<br>
*Datasets: THUMOS14, ActivityNet1.3*<br>
*Task: Temporal Action Localization*

* UBnormal: New Benchmark for Supervised Open-Set Video Anomaly Detection (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Acsintoae_UBnormal_New_Benchmark_for_Supervised_Open-Set_Video_Anomaly_Detection_CVPR_2022_paper.pdf)]
[[Code](https://github.com/lilygeorgescu/UBnormal)]<br>
*Datasets: UBnormal, CHUK, Avenue, Shang-hai Tech*<br>
*Task: Anomaly Detection*

* Open-World Instance Segmentation: Exploiting Pseudo Ground Truth From Learned Pairwise Affinity (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Open-World_Instance_Segmentation_Exploiting_Pseudo_Ground_Truth_From_Learned_Pairwise_CVPR_2022_paper.pdf)]<br>
*Datasets: COCO 17, LVIS, UVO (videos), ADE20k*<br>
*Task: Instance Segmentation*


<!--#### ICLR
#### NeurIPS
#### ICCV
#### ICML
#### IEEE-Access-->
#### ECCV
* Towards Open Set Video Anomaly Detection (ECCV 2022) 
[[Paper](https://arxiv.org/pdf/2208.11113v1.pdf)]<br>
*Datasets: XD Violence, UCF Crime, ShanghaiTech Campus*<br>
*Task: Anomaly Detection*

<!--#### AAAI
#### TPAMI
#### CVPRw
#### WACV
#### IJCV
#### BMVC
#### ICCw
#### Arxiv & Others-->
<!----------------------------------------------------------------------------------------------------------------------------------------------->
### 2021 Papers
#### CVPR
* Generalizing to the Open World: Deep Visual Odometry With Online Adaptation (CVPR 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Generalizing_to_the_Open_World_Deep_Visual_Odometry_With_Online_CVPR_2021_paper.pdf)]<br>
*Datasets: Cityscapes,  KITTI, indoor TUM, NYUv2*<br>
*Task: Depth Estimation*


<!--#### ICLR
#### NeurIPS-->
#### ICCV
* Evidential Deep Learning for Open Set Action Recognition (ICCV 2021) 
[[Paper](https://arxiv.org/pdf/2107.10161v2.pdf)]
[[Code](https://www.rit.edu/actionlab/dear)]<br>
*Datasets: UCF-101, HMDB-51, MiT-v2*<br>
*Task: Action Recognition*

* Unidentified Video Objects: A Benchmark for Dense, Open-World Segmentation (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Unidentified_Video_Objects_A_Benchmark_for_Dense_Open-World_Segmentation_ICCV_2021_paper.pdf)]<br>
*Datasets: UVO, COCO*<br>
*Task: Video Object Detection and Segmentation*

<!-- #### ICML
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
### Older Papers
* Specifying weight priors in bayesian deep neural networks with empirical bayes (AAAI 2020) 
[[Paper](https://arxiv.org/pdf/1906.05323v3.pdf)]<br>
*Datasets: UCF-101, Urban Sound 8K, MNIST, Fashion-MNIST, CIFAR10*<br>
*Task: Image and Audio Classification, Video Activity Recognition*

* P-ODN: prototype-based open Deep network for open Set Recognition (Scientific Reports 2020) 
[[Paper](https://www.nature.com/articles/s41598-020-63649-6)]<br>
*Datasets: UCF11, UCF50, UCF101 and HMDB51*<br>
*Task: Action Recognition*

* Uncertainty-aware audiovisual activity recognition using deep bayesian variational inference (ICCV 2019) 
[[Paper](https://arxiv.org/pdf/1811.10811v3.pdf)]<br>
*Datasets: MiT*<br>
*Task: Audiovisual Action Recognition*

* Bayesian activity recognition using variational inference (NeurIPS 2018) 
[[Paper](https://arxiv.org/pdf/1811.03305v2.pdf)]<br>
*Datasets:  MiT video activity recognition dataset*<br>
*Task: Action Recognition*

* ODN: Opening the deep network for open-set action recognition (ICME 2018) 
[[Paper](https://arxiv.org/pdf/1901.07757v1.pdf)]<br>
*Datasets:  HMDB51, UCF50, UCF101*<br>
*Task: Action Recognition*

<!----------------------------------------------------------------------------------------------------------------------------------------------->
##  Novel Class Discovery Videos
<!----------------------------------------------------------------------------------------------------------------------------------------------->

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
<!----------------------------------------------------------------------------------------------------------------------------------------------->
### 2022 Papers
<!-- #### CVPR
#### ICLR
#### NeurIPS
#### ICCV
#### ICML
#### IEEE-Access-->
#### ECCV
* Text-based Temporal Localization of Novel Events (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740552.pdf)]<br>
*Datasets: Charades-STA Unseen, ActivityNet Captions Unseen*<br>
*Task: Temporal Action Localization*

* Discovering Objects That Can Move (ECCV 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Bao_Discovering_Objects_That_Can_Move_CVPR_2022_paper.pdf)]
[[Code](https://github.com/zpbao/Discovery_Obj_Move)]<br>
*Datasets: KITTI; CATER; TRI-PD*<br>
*Task: Object Segmentation*
<!--#### AAAI
#### TPAMI
#### CVPRw
#### WACV
#### IJCV
#### BMVC
#### ICCw
#### Arxiv & Others-->
<!----------------------------------------------------------------------------------------------------------------------------------------------->
### 2021 Papers
<!-- #### CVPR
#### ICLR
#### NeurIPS-->
#### ICCV
* Joint Representation Learning and Novel Category Discovery on Single- and Multi-Modal Data (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Jia_Joint_Representation_Learning_and_Novel_Category_Discovery_on_Single-_and_ICCV_2021_paper.pdf)]<br>
*Datasets: ImageNet; CIFAR-10/CIFAR-100; Kinetics-400; VGG-Sound*<br>
*Task: Multimodal Data*

* Learning To Better Segment Objects From Unseen Classes With Unlabeled Videos (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Du_Learning_To_Better_Segment_Objects_From_Unseen_Classes_With_Unlabeled_ICCV_2021_paper.pdf)]
[[Code](https://dulucas.github.io/gbopt)]<br>
*Datasets: COCO -> Unseen-VIS; DAVIS*<br>
*Task: Instance Segmentation*
<!--#### ICML
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
## Open Vocabulary Videos
<!----------------------------------------------------------------------------------------------------------------------------------------------->

### 2023 Papers
<!-- #### CVPR-->
#### ICLR
* The Devil is in the Wrongly-classified Samples: Towards Unified Open-set Recognition (ICLR 2023) 
[[Paper](https://openreview.net/pdf?id=xLr0I_xYGAs)]<br>
*Datasets: CIFAR100, LSUN, MiTv2, UCF101, HMDB51*<br>
*Task: Image and Video Classification*
<!--#### NeurIPS
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
<!----------------------------------------------------------------------------------------------------------------------------------------------->
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
## Fine Grained Videos
<!----------------------------------------------------------------------------------------------------------------------------------------------->

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
#### CVPRw-->
#### WACV
* Fine-Grained Activities of People Worldwide (WACV 2023) 
[[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Byrne_Fine-Grained_Activities_of_People_Worldwide_WACV_2023_paper.pdf)]
[[Code](https://visym.github.io/cap)]<br>
*Datasets: Consented Activities of People (CAP)*<br>
*Task: Action Recognition*

* Fine-Grained Affordance Annotation for Egocentric Hand-Object Interaction Videos (WACV 2023) 
[[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Yu_Fine-Grained_Affordance_Annotation_for_Egocentric_Hand-Object_Interaction_Videos_WACV_2023_paper.pdf)]<br>
*Datasets:  EPIC-KITCHENS*<br>
*Task: Action Recognition*
<!--#### IJCV
#### BMVC
#### ICCw
#### Arxiv & Others-->
<!----------------------------------------------------------------------------------------------------------------------------------------------->
### 2022 Papers
#### CVPR

* FineDiving: A Fine-Grained Dataset for Procedure-Aware Action Quality Assessment (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_FineDiving_A_Fine-Grained_Dataset_for_Procedure-Aware_Action_Quality_Assessment_CVPR_2022_paper.pdf)]
[[Code](https://github.com/xujinglin/FineDiving)]<br>
*Datasets: FineDiving*<br>
*Task: Action Quality Assessment*

* Fine-Grained Temporal Contrastive Learning for Weakly-Supervised Temporal Action Localization (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Gao_Fine-Grained_Temporal_Contrastive_Learning_for_Weakly-Supervised_Temporal_Action_Localization_CVPR_2022_paper.pdf)]
[[Code](https://github.com/MengyuanChen21/CVPR2022-FTCL)]<br>
*Datasets: THUMOS14; ActivityNet1.3*<br>
*Task: Temporal Action Localization*

* How Do You Do It? Fine-Grained Action Understanding With Pseudo-Adverbs (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Doughty_How_Do_You_Do_It_Fine-Grained_Action_Understanding_With_Pseudo-Adverbs_CVPR_2022_paper.pdf)]
[[Code](https://github.com/hazeld/PseudoAdverbs)]<br>
*Datasets: VATEX Adverbs, ActivityNet Adverbs and MSR-VTT Adverbs*<br>
*Task: Adverb Recognition*

* EMScore: Evaluating Video Captioning via Coarse-Grained and Fine-Grained Embedding Matching (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Shi_EMScore_Evaluating_Video_Captioning_via_Coarse-Grained_and_Fine-Grained_Embedding_Matching_CVPR_2022_paper.pdf)]
[[Code](https://github.com/shiyaya/emscore)]<br>
*Datasets: VATEX-EVAL; ActivityNet-FOIL *<br>
*Task: Video Captioning*
<!-- #### ICLR
#### NeurIPS
#### ICCV
#### ICML
#### IEEE-Access-->
#### ECCV
* Dynamic Spatio-Temporal Specialization Learning for Fine-Grained Action Recognition (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640381.pdf)]<br>
*Datasets: Diving48*<br>
*Task: Action Recognition*

* Exploring Fine-Grained Audiovisual Categorization with the SSW60 Dataset (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680262.pdf)]
[[Code](https://github.com/visipedia/ssw60)]<br>
*Datasets: SSW60*<br>
*Task: Action Recognition*

* Weakly-Supervised Temporal Action Detection for Fine-Grained Videos with Hierarchical Atomic Actions (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700562.pdf)]
[[Code](https://github.com/lizhi1104/HAAN.git)]<br>
*Datasets: FineAction; FineGym*<br>
*Task: Action Recognition*

* Semantic-Aware Fine-Grained Correspondence (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136910093.pdf)]<br>
*Datasets: DAVIS-2017; JHMDB; Video Instance Parsing (VIP)*<br>
*Task: Video Object Segmentation, Human Pose Tracking, Human Part Tracking*

* Spotting Temporally Precise, Fine-Grained Events in Video (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136950033.pdf)]<br>
*Datasets: Tennis, Figure Skating, FineDiving, and Fine-Gym*<br>
*Task: Temporally Precise Spotting*

* Fine-Grained Egocentric Hand-Object Segmentation: Dataset, Model, and Applications (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890125.pdf)]
[[Code](https://github.com/owenzlz/EgoHOS)]<br>
*Datasets: EPIC-KITCHENS; Ego4d; THU-READ; Escape Room*<br>
*Task: Semantic Segmentation*
<!-- #### AAAI
#### TPAMI-->
#### CVPRw
* FenceNet: Fine-Grained Footwork Recognition in Fencing (CVPRw 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Zhu_FenceNet_Fine-Grained_Footwork_Recognition_in_Fencing_CVPRW_2022_paper.pdf)]<br>
*Datasets:  FFD a publicly available fencing dataset*<br>
*Task: Action Recognition*
<!--#### WACV
#### IJCV
#### BMVC
#### ICCw
#### Arxiv & Others-->
<!----------------------------------------------------------------------------------------------------------------------------------------------->
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
## Long Tail Videos
<!----------------------------------------------------------------------------------------------------------------------------------------------->

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
<!----------------------------------------------------------------------------------------------------------------------------------------------->
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
<!----------------------------------------------------------------------------------------------------------------------------------------------->
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
## Anomaly Detection Videos
<!----------------------------------------------------------------------------------------------------------------------------------------------->
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
<!----------------------------------------------------------------------------------------------------------------------------------------------->
### 2022 Papers
#### CVPR
* UBnormal: New Benchmark for Supervised Open-Set Video Anomaly Detection (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Acsintoae_UBnormal_New_Benchmark_for_Supervised_Open-Set_Video_Anomaly_Detection_CVPR_2022_paper.pdf)]
[[Code](https://github.com/lilygeorgescu/UBnormal)]<br>
*Datasets: UBnormal, CHUK, Avenue, Shang-hai Tech*

* Deep Anomaly Discovery From Unlabeled Videos via Normality Advantage and Self-Paced Refinement (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_Deep_Anomaly_Discovery_From_Unlabeled_Videos_via_Normality_Advantage_and_CVPR_2022_paper.pdf)]
[[Code](https://github.com/lilygeorgescu/AED-SSMTL)]<br>
*Datasets: UCS-Dped1/UCSDped2, Avenue and  ShanghaiTech*

* Self-Supervised Predictive Convolutional Attentive Block for Anomaly Detection (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Ristea_Self-Supervised_Predictive_Convolutional_Attentive_Block_for_Anomaly_Detection_CVPR_2022_paper.pdf)]
[[Code](https://github.com/ristea/sspcab)]<br>
*Datasets: MVTec AD, Avenue and  ShanghaiTech*

* Anomaly Detection via Reverse Distillation From One-Class Embedding (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Deng_Anomaly_Detection_via_Reverse_Distillation_From_One-Class_Embedding_CVPR_2022_paper.pdf)]<br>
*Datasets: MVTec; MNIST, FashionMNIST and CIFAR10*<br>

* Bayesian Nonparametric Submodular Video Partition for Robust Anomaly Detection (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Sapkota_Bayesian_Nonparametric_Submodular_Video_Partition_for_Robust_Anomaly_Detection_CVPR_2022_paper.pdf)]<br>
*Datasets: ShanghaiTech, Avenue, UCF-Crime*

* Towards Total Recall in Industrial Anomaly Detection (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf)]
[[Code](https://github.com/amazon-science/patchcore-inspection)]<br>
*Datasets: MVTec; Magnetic Tile Defects (MTD); Mini Shanghai Tech Campus(mSTC)*


* Generative Cooperative Learning for Unsupervised Video Anomaly Detection (CVPR 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zaheer_Generative_Cooperative_Learning_for_Unsupervised_Video_Anomaly_Detection_CVPR_2022_paper.pdf)]
[[Code](https://github.com/amazon-science/patchcore-inspection)]<br>
*Datasets: UCF-Crime (UCFC); ShanghaiTech*

<!-- #### ICLR
#### NeurIPS
#### ICCV
#### ICML
#### IEEE-Access-->
#### ECCV
* Towards Open Set Video Anomaly Detection (ECCV 2022) 
[[Paper](https://arxiv.org/pdf/2208.11113v1.pdf)]<br>
*Datasets: XD Violence, UCF Crime, ShanghaiTech Campus*

* Scale-Aware Spatio-Temporal Relation Learning for Video Anomaly Detection (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640328.pdf)]
[[Code](https://github.com/nutuniv/SSRL)]<br>
*Datasets: UCF-Crime (UCFC); ShanghaiTech*

* Dynamic Local Aggregation Network with Adaptive Clusterer for Anomaly Detection (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640398.pdf)]
[[Code](https://github.com/Beyond-Zw/DLAN-AC)]<br>
*Datasets: CUHK Avenue; UCSD Ped2; ShanghaiTech*

* Video Anomaly Detection by Solving Decoupled Spatio-Temporal Jigsaw Puzzles (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700490.pdf)]<br>
*Datasets: CUHK Avenue; UCSD Ped2; ShanghaiTech*

* Self-Supervised Sparse Representation for Video Anomaly Detection (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730727.pdf)]
[[Code](https://github.com/louisYen/S3R)]<br>
*Datasets: ShanghaiTech, UCF-Crime, and XD-Violence*

* Registration Based Few-Shot Anomaly Detection (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840300.pdf)]
[[Code](https://github.com/MediaBrain-SJTU/RegAD)]<br>
*Datasets: MVTec; MPDD*

* DenseHybrid: Hybrid Anomaly Detection for Dense Open-set Recognition (ECCV 2022) 
[[Paper](https://arxiv.org/pdf/2207.02606v1.pdf)]
[[Code](https://github.com/matejgrcic/DenseHybrid)]<br>
*Datasets: Fishyscapes, SegmentMeIfYouCan (SMIYC), StreetHazards*

<!--#### AAAI
#### TPAMI-->
#### CVPRw
* Unsupervised Anomaly Detection From Time-of-Flight Depth Images (CVPRw 2022) 
[[Paper](https://arxiv.org/abs/2203.01052v2)]<br>
*Datasets: TIMo*<br>

* Adversarial Machine Learning Attacks Against Video Anomaly Detection Systems (CVPRw 2022) 
[[Paper](https://arxiv.org/abs/2204.03141v1)]<br>
*Datasets: CUHK Avenue, the ShanghaiTech Campus*<br>

* Anomaly Detection in Autonomous Driving: A Survey (CVPRw 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022W/WAD/papers/Bogdoll_Anomaly_Detection_in_Autonomous_Driving_A_Survey_CVPRW_2022_paper.pdf)]

* AnoDDPM: Anomaly Detection With Denoising Diffusion Probabilistic Models Using Simplex Noise (CVPRw 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Wyatt_AnoDDPM_Anomaly_Detection_With_Denoising_Diffusion_Probabilistic_Models_Using_Simplex_CVPRW_2022_paper.pdf)]
[[Code](https://github.com/Julian-Wyatt/AnoDDPM)]<br>
*Datasets: MVTec AD*<br>

<!--#### WACV
#### IJCV
#### BMVC
#### ICCw
#### Arxiv & Others-->
<!----------------------------------------------------------------------------------------------------------------------------------------------->
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
