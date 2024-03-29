# Open World Papers

Current machine learning algorithms are bound to a closed-world assumption. This means that these models assume the number of categories the model will encounter in the test time is predefined. However, this assumption needs to be revised in the real world. There are many challenges associated with the real-world setting that traditional machine learning models can not resolve. The uncertainty a model can encounter during test time and the strategy to address it have been considered recently under different computer vision branches. 
In tackling the real world, we want to consider layers of uncertainty. In the following sections, we defined each challenge and what field of data will create. We created this repo to list current methods used in recent years to address uncertainty (specifically novel class and open-set) in the world in contemporary top venues like **CVPR, CVPRw, NeurIPS, ICCV, ICCVw, ECCV, ECCVw, ICLR, ICML, BMVC, WACV, WACVw, TPAMI, AAAI,** and relevant papers from **Arxiv** and other venues. 

Finally, since our primary focus is fine-grained or long-tailed novel action discovery, we also address related works in this area. 
Without further due, let us dive into the fantastic and challenging world of uncertainty, unseens, and unknowns. 

**Disclaimer:** 
Since some papers belong to multiple categories, I put them in all related categories; some papers got accepted in conferences or journals later, so there might be both an arxiv or submission form of the paper in the repository. Usually, it means where I found the paper, not where it has been published. I made sure that there were no mistakes, but there might be a few, so if you noticed that I missed something, make a pull request.


## Contents
- [Introduction](#Introduction)
  - [Unseen Environments](#Unseen-Environments)
  - [Unseen Categories](#Unseen-Categories)
  - [Unknown Categories](#Unknown-Categories)
- [Zero-Shot Learning](#Zero-Shot-Learning)  
- [Out-of-Distribution Detection](#Out-of-Distribution-Detection) 
- [Open-Set Recognition](#Open-Set-Recognition)
- [Novel Class Discovery](#Novel-Class-Discovery))
- [Open Vocabulary](#Open-Vocabulary)
- [Fine Grained](#Fine-Grained)
- [Long Tail](#Long-Tail)
- [Anomaly Detection](#Anomaly-Detection)
- [Video Open World Papers](#Video-Open-World-Papers)
- [Zero-Shot Learning Videos](#Zero-Shot-Learning-Videos)  
- [Out-of-Distribution Detection Videos](#Out-of-Distribution-Detection-Videos) 
- [Open-Set Recognition Videos](#Open-Set-Recognition-Videos)
- [Novel Class Discovery Videos](#Novel-Class-Discovery-Videos)
- [Open Vocabulary Videos](#Open-Vocabulary-Videos)
- [Fine Grained Videos](#Fine-Grained-Videos)
- [Long Tail Videos](#Long-Tail-Videos)
- [Anomaly Detection Videos](#Anomaly-Detection-Videos)
- [Novelty Detection](#Novelty-Detection)
- [Other Related Papers](#Other-Related-Papers)
- [Action Recognition Related](#Action-Recognition-Related)

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

In this repository section [Zero-Shot Learning](#Zero-Shot-Learning) and [Zero-Shot Learning Videos](#Zero-Shot-Learning-Videos) has summarized the recent papers for both *Zero-shot* and *Generalized Zero-shot*.

### Unknown Categories
Telsa scientists are relieved that their autopilot model is ready to be employed in the real world. Their model can handle unseen environments and unseen actions and categories. However, there is still a minor problem; there is one extremely likely prediction that one day their autopilot will encounter objects or actions, they are unaware of at training time. So they decide to train their model to make them aware anytime it sees something new or rejects it. First, however, they need to learn how to regulate this uncertainty. Some scientists suggest that the model rejects anything from the training time distribution. In computer vision, this problem is known as [Out-of-Distribution Detection](#Out-of-Distribution-Detection) or its video version as [Out-of-Distribution Detection Videos](#Out-of-Distribution-Detection-Videos). Another highly overlapped scenario is [Anomaly Detection](#Anomaly-Detection) or its video version [Anomaly Detection Videos](#Anomaly-Detection-Videos). In Anomaly detection, the anomal object or events usually is rare, and we are interested in extracting them from data for OOD. However, the rejected samples are typically abundant, and the rejection can be done automatically. 

Sometimes it is essential to reject data that we need to know their category while they could be in distribution. This problem is called [Open-Set Recognition](#Open-Set-Recognition), and its video [Open-Set Recognition Videos](#Open-Set-Recognition-Videos). 

The problem with OOD or OSR is that the model rejects the unknowns, so though it could improve itself, it does not learn the new categories, which is in direct contrast with humans, which start which categories of new objects or actions. So if an agent can detect these novel categories and cluster them, it could use this novel knowledge to its benefit and use from the vast number of novel but unlabelled samples. This problem is called [Novel Class Discovery](#Novel-Class-Discovery) or its video version [Novel Class Discovery Videos](#Novel-Class-Discovery-Videos). Unfortunately, when there is a mixture of unknown and known categories in test time, the model is prone to assign samples mostly to either known or unknown categories. **Generalized Category Discovery** addresses this problem when encountering novel classes alongside already seen classes in the test time. 

While novel class discovery and its cousin generalized category discovery have a lot of challenges, dealing with fine-grained categories where the categories have subtle differences and may have a lot of overlap or long-tail distribution where the frequency of each category can differ drastically elevates these obstacles to a very dark realm. So in this repository, we also consider the works that are done in [Fine Grained](Fine-Grained.md#Fine-Grained) or [Fine Grained Videos](#Fine-Grained-Videos). While Long-tail is still in its infancy, especially for the video field, the challenge it introduces to the open world is very real, therefore works that focus on [Long Tail](#Long-Tail) or [Long Tail Videos](#Long-Tail-Videos) also has been collected. 

For more concise terminology, we use **claster** when referring to clusters depicting an already seen class or an unseen proposed class. 

| **Problem Name**  |     **Problem Goal** |
|---------------|-----------|
| Zero-shot Learning| Classify test samples from a set of unseen but known categories.| 
| Generalized Zero-shot Learning| Classify test samples from a set of seen and unseen but known categories.| 
| Open-set Recognition| Classify test samples from a set of seen categories while rejecting samples from unknown categories.|
| Out-of-Distribution Detection| Classify test samples from a set of seen categories while rejecting samples which are not from the training distribution.|
| Novel Class Discovery| Classify test samples from a set of unknown categories into proposed clasters.|
| Generalized Category Discovery| Classify test samples from a set of seen or unknown categories into seen categories or proposed clasters.|
| Open Vocabulary| Classify test samples from a set of seen or unknown categories into proposed clasters and find the corresponding name for that claster with the help of additional information like another modality or language models.|
| Fine Grained| Classify test samples from a set of categories that have very subtle differences.|
| Long Tailed| Classify test samples from a set of categories that have very different frequencies often in form of a power law distribution.|

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
### Surveys
* Zero-shot and Few-shot Learning with Knowledge Graphs: A Comprehensive Survey (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2112.10006)]

* A Survey of Zero-shot Generalisation in Deep Reinforcement Learning (JAIR 2023) 
[[Paper](https://arxiv.org/abs/2111.09794)]

* Survey of Visual-Semantic Embedding Methods for Zero-Shot Image Retrieval (ICMLA 2021) 
[[Paper](https://arxiv.org/abs/2105.07391)]

* Knowledge-aware Zero-Shot Learning: Survey and Perspective (IJCAI 2021 Survey Track) 
[[Paper](https://arxiv.org/abs/2103.00070)]

* A Survey of Deep Learning for Low-Shot Object Detection (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2112.02814)]

* Weak Novel Categories without Tears: A Survey on Weak-Shot Learning (Arxiv 2021) 
[[Paper](https://arxiv.org/abs/2110.02651)]

### 2023 Papers
#### CVPR

* Learning Attention as Disentangler for Compositional Zero-shot Learning (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.15111)]

* Delving into Shape-aware Zero-shot Semantic Segmentation (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2304.08491)]

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
#### CVPRw-->
#### WACV
* Zero-Shot Versus Many-Shot: Unsupervised Texture Anomaly Detection (WACV 2023) 
[[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Aota_Zero-Shot_Versus_Many-Shot_Unsupervised_Texture_Anomaly_Detection_WACV_2023_paper.pdf)]
[[Code](https://drive.google.com/drive/folders/10OyPzvI3H6llCZBxKxFlKWt1Pw1tkMK1)]<br>
*Datasets: MVTec*<br>
*Task: Texture Anomaly Detection*

* Learning Attention Propagation for Compositional Zero-Shot Learning (WACV 2023) 
[[Paper](http://arxiv.org/abs/2210.11557)]<br>
*Datasets:  MIT-States, CGQA, UT-Zappos*<br>
*Task: Image Classification*

* InDiReCT: Language-Guided Zero-Shot Deep Metric Learning for Images (WACV 2023) 
[[Paper](http://arxiv.org/abs/2211.12760)]<br>
*Datasets: Synthetic Cars, Cars196*<br>
*Task: Deep Metric Learning Images*


<!--#### IJCV
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

* PatchComplete: Learning Multi-Resolution Patch Priors for 3D Shape Completion on Unseen Categories (NeurIPS 2022) 
[[Paper](https://arxiv.org/abs/2206.04916)]
[[Code](https://yuchenrao.github.io/projects/patchComplete/patchComplete.html)]<br>
*Datasets: ShapeNet, ScanNet, Scan2CAD*<br>
*Task: 3D Shape Reconstruction*

* Mining Unseen Classes via Regional Objectness: A Simple Baseline for Incremental Segmentation (NeurIPS 2022) 
[[Paper](https://arxiv.org/abs/2211.06866)]
[[Code](https://github.com/zkzhang98/MicroSeg)]<br>
*Datasets: Pascal VOC and ADE20K*<br>
*Task: Continual Image Classification*

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

#### AAAI
* Open Vocabulary Electroencephalography-to-Text Decoding and Zero-Shot Sentiment Classification (AAAI 2022) 
[[Paper](https://arxiv.org/abs/2112.02690)]
[[Code](https://github.com/MikeWangWZHL/EEG-To-Text)]<br>
*Datasets:  ZuCo*<br>
*Task: Brain Signals Language Decoding*
<!--#### TPAMI-->
#### CVPRw
* Semantically Grounded Visual Embeddings for Zero-Shot Learning (CVPRw 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022W/MULA/papers/Nawaz_Semantically_Grounded_Visual_Embeddings_for_Zero-Shot_Learning_CVPRW_2022_paper.pdf)]<br>
*Datasets: CUB(312−d), AWA(85−d) and aPY(64−d); FLO*<br>
*Task: Semantic Embeddings*

* Zero-Shot Learning Using Multimodal Descriptions (CVPRw 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022W/L3D-IVU/papers/Mall_Zero-Shot_Learning_Using_Multimodal_Descriptions_CVPRW_2022_paper.pdf)]<br>
*Datasets: CUB-200-2011 (CUB), SUN attributes (SUN) and DeepFashion (DF)*<br>
*Task: Multimodal Zero-Shot*

#### WACV

* COCOA: Context-Conditional Adaptation for Recognizing Unseen Classes in Unseen Domains (WACV 2022) 
[[Paper](https://openaccess.thecvf.com/content/WACV2022/papers/Mangla_COCOA_Context-Conditional_Adaptation_for_Recognizing_Unseen_Classes_in_Unseen_Domains_WACV_2022_paper.pdf)]<br>
*Datasets: DomainNet, DomainNet-LS*<br>
*Task: Domain Generalization and Novel Class Discovery*
<!--#### IJCV
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
#### IJCV-->
#### BMVC

* Structured Latent Embeddings for Recognizing Unseen Classes in Unseen Domains (BMVC 2021) 
[[Paper](https://www.bmvc2021-virtualconference.com/assets/papers/1377.pdf)]<br>
*Datasets: DomainNet, DomainNet-LS*<br>
*Task: Domain Generalization*

<!--#### ICCw
#### Arxiv & Others-->
### Older Papers

* Stacked Semantics-Guided Attention Model for Fine-Grained Zero-Shot Learning (NeurIPS 2018) 
[[Paper](https://papers.nips.cc/paper/2018/hash/9087b0efc7c7acd1ef7e153678809c77-Abstract.html)]<br>
*Datasets: CUB and NABird*<br>
*Task: Image Classification*

* MSplit LBI: Realizing Feature Selection and Dense Estimation Simultaneously in Few-shot and Zero-shot Learning (ICML 2018) 
[[Paper](http://proceedings.mlr.press/v80/zhao18c.html)]<br>
*Datasets: Animals with Attributes (AwA), Caltech-UCSD Birds-200-2011 (CUB) and ImageNet 2012/2010*<br>
*Task: Image Classification*

* A Boundary Based Out-of-Distribution Classifier for Generalized Zero-Shot Learning (ECCV 2020) 
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690562.pdf)]<br>
*Datasets: AWA1, AWA2, CUB, FLO and SUN*<br>
*Task: Out-of-Distribution Image Classification*

* Towards Recognizing Unseen Categories in Unseen Domains (ECCV 2020) 
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680460.pdf)]
[[Code](https://github.com/mancinimassimiliano/CuMix)]<br>
*Datasets: AWA, CUB, FLO and SUN, PACS*<br>
*Task: Out-of-Distribution Image Classification*

* Zero-Shot Visual Imitation (ICLR 2018 Oral) 
[[Paper](https://openreview.net/forum?id=BkisuzWRW)]
[[Code](https://github.com/pathak22/zeroshot-imitation)]<br>
*Datasets: Rope manipulation using Baxter robot, Navigation of a wheeled robot in cluttered office environments, Simulated 3D navigation*<br>
*Task: Imitation Learning*

* Generalized Zero-shot Learning using Open Set Recognition (BMVC 2019) 
[[Paper](https://bmvc2019.org/wp-content/papers/0035.html)]<br>
*Datasets: AWA1, APY, FLO, and CUB*<br>
*Task: Image Classification*

* Image Captioning with Unseen Objects (BMVC 2019) 
[[Paper](https://bmvc2019.org/wp-content/papers/0124.html)]<br>
*Datasets: COCO*<br>
*Task: Image Captioning*

* Safe Deep Semi-Supervised Learning for Unseen-Class Unlabeled Data (ICML 2020) 
[[Paper](https://proceedings.mlr.press/v119/guo20i.html)]
[[Code](http://www.lamda.nju.edu.cn/code_DS3L.ashx)]<br>
*Datasets: MNIST, CIFAR10*<br>
*Task: Image Classification*

* Hallucinative Topological Memory for Zero-Shot Visual Planning (ICML 2020) 
[[Paper](https://arxiv.org/abs/2002.12336)]<br>
*Datasets: Mujoco simulation (Block wall, Block  wall  with  complex  obstacle, Block insertion, Robot  manipulation)*<br>
*Task: Visual Planning*

* “Other-Play” for Zero-Shot Coordination (ICML 2020) 
[[Paper](https://arxiv.org/abs/2003.02979)]
[[Code](https://bit.ly/2vYkfI7)]<br>
*Datasets: “lever game”,  Hanabi with AI Agents*<br>
*Task: Zero-Shot Coordination*

* Discovering Human Interactions With Novel Objects via Zero-Shot Learning (CVPR 2020) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Discovering_Human_Interactions_With_Novel_Objects_via_Zero-Shot_Learning_CVPR_2020_paper.pdf)]
[[Code](https://github.com/scwangdyd/zero_shot_hoi)]<br>
*Datasets: V-COCO, HICO-DET*<br>
*Task: Human Object Interaction*

* Locality and Compositionality in Zero-Shot Learning (ICLR 2020) 
[[Paper](https://openreview.net/forum?id=Hye_V0NKwr)]<br>
*Datasets: AwA2, CUB-200-2011*<br>
*Task: Image Classification*

* Learning to Group: A Bottom-Up Framework for 3D Part Discovery in Unseen Categories (ICLR 2020) 
[[Paper](https://openreview.net/forum?id=rkl8dlHYvB)]
[[Code](https://github.com/tiangeluo/Learning-to-Group)]<br>
*Datasets: PartNet*<br>
*Task: 3D Part Discovery*

<!----------------------------------------------------------------------------------------------------------------------------------------------->
## Out-of-Distribution Detection
<!----------------------------------------------------------------------------------------------------------------------------------------------->
#### Surveys

* A Comprehensive Review of Trends, Applications and Challenges In Out-of-Distribution Detection (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2209.12935)]

* Out-Of-Distribution Generalization on Graphs: A Survey (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2202.07987)]

* A Unified Survey on Anomaly, Novelty, Open-Set, and Out-of-Distribution Detection: Solutions and Future Challenges (TMLR 2023) 
[[Paper](https://arxiv.org/abs/2110.14051)]

* Generalized Out-of-Distribution Detection: A Survey (Arxiv 2021) 
[[Paper](https://arxiv.org/abs/2110.11334)]

* Towards Out-Of-Distribution Generalization: A Survey (Arxiv 2021) 
[[Paper](https://arxiv.org/abs/2108.13624)]

* A Survey on Assessing the Generalization Envelope of Deep Neural Networks: Predictive Uncertainty, Out-of-distribution and Adversarial Samples (Arxiv 2020) 
[[Paper](https://arxiv.org/abs/2008.09381)]

### 2023 Papers
#### CVPR
* Mind the Label Shift of Augmentation-based Graph OOD Generalization (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.14859)]

* Balanced Energy Regularization Loss for Out-of-distribution Detection (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Choi_Balanced_Energy_Regularization_Loss_for_Out-of-Distribution_Detection_CVPR_2023_paper.html)]

* NICO++: Towards better bechmarks for Out-of-Distribution Generalization (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2204.08040)]

* LINe: Out-of-Distribution Detection by Leveraging Important Neurons (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.13995)]

* Detection of out-of-distribution samples using binary neuron activation patterns (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2212.14268)]

* Decoupling MaxLogit for Out-of-Distribution Detection (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Decoupling_MaxLogit_for_Out-of-Distribution_Detection_CVPR_2023_paper.html)]

* Rethinking Out-of-Distribution Detection: Masked Image Modeling is All You Need (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2302.02615)]

* Out-of-Distributed Semantic Pruning for Robust Semi-Supervised Learning (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Out-of-Distributed_Semantic_Pruning_for_Robust_Semi-Supervised_Learning_CVPR_2023_paper.html)]

* Devil is in the Queries: Advancing Mask Transformers for Real-world Medical Image Segmentation and Out-of-Distribution Localization (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2304.00212)]

* GEN: Pushing the Limits of Softmax-Based Out-of-Distribution Detection (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_GEN_Pushing_the_Limits_of_Softmax-Based_Out-of-Distribution_Detection_CVPR_2023_paper.html)]

* Are Data-driven Explanations Robust against Out-of-distribution Data? (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.16390)]

* Uncertainty-Aware Optimal Transport for Semantically Coherent Out-of-Distribution Detection (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.10449)]

* Distribution Shift Inversion for Out-of-Distribution Prediction (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_Distribution_Shift_Inversion_for_Out-of-Distribution_Prediction_CVPR_2023_paper.html)]

* PoseExaminer: Automated Testing of Out-of-Distribution Robustness in Human Pose and Shape Estimation (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.07337)]

* Block Selection Method for Using Feature Norm in Out-of-Distribution Detection (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2212.02295)]


#### ICLR
* Offline RL with No OOD Actions: In-Sample Learning via Implicit Value Regularization (ICLR 2023 top 5%) 
[[Paper](https://openreview.net/forum?id=boNyg20-JDm)]

* Harnessing Out-Of-Distribution Examples via Augmenting Content and Style (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=boNyg20-JDm)]<br>
*Datasets:  SVHN, CIFAR10, LSUN, DTD, CUB, Flowers, Caltech, Dogs*<br>

* Out-of-Distribution Detection and Selective Generation for Conditional Language Models (ICLR 2023 top 25%) 
[[Paper](https://openreview.net/forum?id=kJUS5nD0vPB)]

* A framework for benchmarking Class-out-of-distribution detection and its application to ImageNet (ICLR 2023 top 25%) 
[[Paper](https://openreview.net/forum?id=Iuubb9W6Jtk)]

* Turning the Curse of Heterogeneity in Federated Learning into a Blessing for Out-of-Distribution Detection (ICLR 2023 top 25%) 
[[Paper](https://openreview.net/forum?id=mMNimwRb7Gr)]

* Modeling the Data-Generating Process is Necessary for Out-of-Distribution Generalization (ICLR 2023 top 25%) 
[[Paper](https://openreview.net/forum?id=uyqks-LILZX)]

* Extremely Simple Activation Shaping for Out-of-Distribution Detection (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=ndYXTEL6cZz)]

* The Tilted Variational Autoencoder: Improving Out-of-Distribution Detection (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=YlGsTZODyjz)]

* Pareto Invariant Risk Minimization: Towards Mitigating the Optimization Dilemma in Out-of-Distribution Generalization (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=esFxSb_0pSL)]

* Out-of-Distribution Detection based on In-Distribution Data Patterns Memorization with Modern Hopfield Energy (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=KkazG4lgKL)]

* Improving Out-of-distribution Generalization with Indirection Representations (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=0f-0I6RFAch)]

* Out-of-distribution Detection with Implicit Outlier Transformation (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=hdghx6wbGuD)]

* Topology-aware Robust Optimization for Out-of-Distribution Generalization (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=ylMq8MBnAp)]

* Out-of-distribution Representation Learning for Time Series Classification (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=gUZWOE42l6Q)]

* How to Exploit Hyperspherical Embeddings for Out-of-Distribution Detection? (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=aEFaE0W5pAd)]

* Energy-based Out-of-Distribution Detection for Graph Neural Networks (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=zoz7Ze4STUL)]

* Don’t forget the nullspace! Nullspace occupancy as a mechanism for out of distribution failure (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=39z0zPZ0AvB)]

* InPL: Pseudo-labeling the Inliers First for Imbalanced Semi-supervised Learning (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=m6ahb1mpwwX)]

* Diversify and Disambiguate: Out-of-Distribution Robustness via Disagreement (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=RVTOp3MwT3n)]

* On the Effectiveness of Out-of-Distribution Data in Self-Supervised Long-Tail Learning. (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=v8JIQdiN9Sh)]

* Can Agents Run Relay Race with Strangers? Generalization of RL to Out-of-Distribution Trajectories (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=ipflrGaf7ry)]


<!--#### NeurIPS
#### ICCV
#### ICML
#### IEEE-Access
#### ECCV
#### AAAI
#### TPAMI
#### CVPRw-->
#### WACV

* Mixture Outlier Exposure: Towards Out-of-Distribution Detection in Fine-Grained Environments (WACV 2023) 
[[Paper](http://arxiv.org/abs/2106.03917)]
[[Code](https://github.com/zjysteven/MixOE)]<br>
*Datasets: WebVision 1.0*<br>
*Task: Out-of-Distribution Detection Images*

* Out-of-Distribution Detection via Frequency-Regularized Generative Models (WACV 2023) 
[[Paper](http://arxiv.org/abs/2208.09083)]
[[Code](https://github.com/mu-cai/FRL)]<br>
*Datasets: CIFAR-10, Fashion-MNIST (ID), SVHN, LSUN, MNIST, KMNIST, Omniglot, NotMNIST, Noise, Constant*<br>
*Task: Out-of-Distribution Detection Images*

* Hyperdimensional Feature Fusion for Out-of-Distribution Detection (WACV 2023) 
[[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Wilson_Hyperdimensional_Feature_Fusion_for_Out-of-Distribution_Detection_WACV_2023_paper.pdf)]
[[Code](https://github.com/SamWilso/HDFF_Official)]<br>
*Datasets: CIFAR10 and CIFAR100 (ID), iSUN, TinyImageNet (croppedand resized: TINc and TINr), LSUN (cropped and resized: LSUNc and LSUNr), SVHN, MNIST, KMNIST, FashionMNIST, Textures*<br>
*Task: Out-of-Distribution Detection Images*

* Out-of-Distribution Detection With Reconstruction Error and Typicality-Based Penalty (WACV 2023) 
[[Paper](http://arxiv.org/abs/2212.12641)]<br>
*Datasets: CIFAR-10, TinyImageNet, and ILSVRC2012*<br>
*Task: Out-of-Distribution Detection Image Reconstruction*

* Heatmap-Based Out-of-Distribution Detection (WACV 2023) 
[[Paper](http://arxiv.org/abs/2211.08115)]
[[Code](https://github.com/jhornauer/heatmap_ood)]<br>
*Datasets: CIFAR-10, CIFAR-100 and Tiny ImageNet*<br>
*Task: Out-of-Distribution Detection Images*


<!--#### IJCV
#### BMVC
#### ICCw-->
#### Arxiv & Others
* VRA: Out-of-Distribution Detection with variational rectified activations (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2302.11716)]
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

#### ICLR

* Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution (ICLR 2022 Oral) 
[[Paper](https://openreview.net/forum?id=UYneFzXSJWh)]
[[Code](https://github.com/AnanyaKumar/transfer_learning)]<br>
*Datasets:  DomainNet, BREEDS-Living-17, BREEDS-Entity-30, CIFAR-10→STL, CIFAR-10→CIFAR-10.1, ImageNet-1K — where the OODtest sets are ImageNetV2, ImageNet-R, ImageNet-A, and ImageNet-Sketch —, FMoW Geo-shift*<br>

* Vision-Based Manipulators Need to Also See from Their Hands (ICLR 2022 Oral) 
[[Paper](https://openreview.net/forum?id=RJkAHKp7kNZ)]
[[Code](https://sites.google.com/view/seeing-from-hands)]<br>
*Datasets: PyBullet physics engine, Meta-World*<br>

* Asymmetry Learning for Counterfactually-invariant Classification in OOD Tasks (ICLR 2022 Oral) 
[[Paper](https://openreview.net/forum?id=avgclFZ221l)]<br>
*Datasets: MNIST-{3,4},...*<br>

* Poisoning and Backdooring Contrastive Learning (ICLR 2022 Oral) 
[[Paper](https://openreview.net/forum?id=iC4UHbQ01Mp)]<br>
*Datasets: Conceptual Captions dataset*<br>

* Representational Continuity for Unsupervised Continual Learning (ICLR 2022 Oral) 
[[Paper](https://openreview.net/forum?id=9Hrka5PA7LW)]
[[Code](https://github.com/divyam3897/UCL)]<br>
*Datasets: Split CIFAR-10, Split CIFAR-100, Split Tiny-ImageNet*<br>

* Natural Posterior Network: Deep Bayesian Predictive Uncertainty for Exponential Family Distributions (ICLR 2022 Spotlight) 
[[Paper](https://openreview.net/forum?id=tV3N0DWMxCg)]<br>
*Datasets: Sensorless Drive, MNIST, FMNIST, CIFAR-10*<br>

* Pessimistic Bootstrapping for Uncertainty-Driven Offline Reinforcement Learning (ICLR 2022 Spotlight) 
[[Paper](https://openreview.net/forum?id=Y4cs1Z3HnqL)]
[[Code](https://github.com/Baichenjia/PBRL)]<br>
*Datasets: Gym*<br>

* Compositional Attention: Disentangling Search and Retrieval (ICLR Spotlight 2022) 
[[Paper](https://openreview.net/forum?id=IwJPj2MBcIa)]
[[Code](https://github.com/sarthmit/Compositional-Attention)]<br>
*Datasets: Sort-of-CLEVR, CIFAR10, FashionMNIST, SVHN, Equilateral Triangle Detection*<br>

* Asymmetry Learning for Counterfactually-invariant Classification in OOD Tasks? (ICLR 2022 Oral)
[[Paper](https://openreview.net/forum?id=avgclFZ221l)]

* Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution (ICLR 2022 Oral)
[[Paper](https://openreview.net/forum?id=UYneFzXSJWh)]

* Vision-Based Manipulators Need to Also See from Their Hands (ICLR 2022 Oral)
[[Paper](https://openreview.net/forum?id=RJkAHKp7kNZ)]

* Uncertainty Modeling for Out-of-Distribution Generalization (ICLR 2022)
[[Paper](https://openreview.net/forum?id=6HN7LHyzGgC)]

* Igeood: An Information Geometry Approach to Out-of-Distribution Detection (ICLR 2022)
[[Paper](https://openreview.net/forum?id=mfwdY3U_9ea)]

* Revisiting flow generative models for Out-of-distribution detection (ICLR 2022)
[[Paper](https://openreview.net/forum?id=6y2KBh-0Fd9)]

* Invariant Causal Representation Learning for Out-of-Distribution Generalization (ICLR 2022)
[[Paper](https://openreview.net/forum?id=-e4EXDWXnSn)]

* PI3NN: Out-of-distribution-aware Prediction Intervals from Three Neural Networks (ICLR 2022)
[[Paper](https://openreview.net/forum?id=NoB8YgRuoFU)]

* A Statistical Framework for Efficient Out of Distribution Detection in Deep Neural Networks (ICLR 2022)
[[Paper](https://openreview.net/forum?id=Oy9WeuZD51)]

* Leveraging unlabeled data to predict out-of-distribution performance (ICLR 2022)
[[Paper](https://openreview.net/forum?id=o_HsiMPYh_x)]

* Out-of-distribution Generalization in the Presence of Nuisance-Induced Spurious Correlations (ICLR 2022)
[[Paper](https://openreview.net/forum?id=12RoR2o32T)]

* The Role of Pretrained Representations for the OOD Generalization of RL Agents (ICLR 2022)
[[Paper](https://openreview.net/forum?id=8eb12UQYxrG)]

#### NeurIPS
* Is Out-of-Distribution Detection Learnable? (NeurIPS 2022 Outstanding) 
[[Paper](https://neurips.cc/virtual/2022/poster/55375)]

* Towards Out-of-Distribution Sequential Event Prediction: A Causal Treatment (NeurIPS 2022) 
[[Paper](https://neurips.cc/virtual/2022/poster/54274)]

* GraphDE: A Generative Framework for Debiased Learning and Out-of-Distribution Detection on Graphs (NeurIPS 2022) 
[[Paper](https://neurips.cc/virtual/2022/poster/54922)]

* Provably Adversarially Robust Detection of Out-of-Distribution Data (Almost) for Free (NeurIPS 2022) 
[[Paper](https://neurips.cc/virtual/2022/poster/53127)]

* Your Out-of-Distribution Detection Method is Not Robust! (NeurIPS 2022) 
[[Paper](https://neurips.cc/virtual/2022/poster/53795)]

* Density-driven Regularization for Out-of-distribution Detection (NeurIPS 2022) 
[[Paper](https://neurips.cc/virtual/2022/poster/53993)]

* GOOD: A Graph Out-of-Distribution Benchmark (NeurIPS 2022) 
[[Paper](https://neurips.cc/virtual/2022/poster/55695)]

* Learning Substructure Invariance for Out-of-Distribution Molecular Representations (NeurIPS 2022) 
[[Paper](https://neurips.cc/virtual/2022/poster/55440)]

* Assaying Out-Of-Distribution Generalization in Transfer Learning (NeurIPS 2022) 
[[Paper](https://neurips.cc/virtual/2022/poster/53190)]

* Functional Indirection Neural Estimator for Better Out-of-distribution Generalization (NeurIPS 2022) 
[[Paper](https://neurips.cc/virtual/2022/poster/54691)]

* ZooD: Exploiting Model Zoo for Out-of-Distribution Generalization (NeurIPS 2022) 
[[Paper](https://neurips.cc/virtual/2022/poster/54155)]

* Multi-Instance Causal Representation Learning for Instance Label Prediction and Out-of-Distribution Generalization (NeurIPS 2022) 
[[Paper](https://neurips.cc/virtual/2022/poster/55138)]

* RankFeat: Rank-1 Feature Removal for Out-of-distribution Detection (NeurIPS 2022) 
[[Paper](https://neurips.cc/virtual/2022/poster/55298)]

* Delving into Out-of-Distribution Detection with Vision-Language Representations (NeurIPS 2022) 
[[Paper](https://neurips.cc/virtual/2022/poster/54703)]

* Learning Invariant Graph Representations for Out-of-Distribution Generalization (NeurIPS 2022) 
[[Paper](https://neurips.cc/virtual/2022/poster/55398)]

* OpenOOD: Benchmarking Generalized Out-of-Distribution Detection (NeurIPS 2022) 
[[Paper](https://neurips.cc/virtual/2022/poster/55763)]

* Out-of-Distribution Detection with An Adaptive Likelihood Ratio on Informative Hierarchical VAE (NeurIPS 2022) 
[[Paper](https://neurips.cc/virtual/2022/poster/54902)]

* Boosting Out-of-distribution Detection with Typical Features (NeurIPS 2022) 
[[Paper](https://neurips.cc/virtual/2022/poster/55010)]

* Learning Causally Invariant Representations for Out-of-Distribution Generalization on Graphs (NeurIPS 2022) 
[[Paper](https://neurips.cc/virtual/2022/poster/54643)]

* Evaluating Out-of-Distribution Performance on Document Image Classifiers (NeurIPS 2022) 
[[Paper](https://neurips.cc/virtual/2022/poster/55621)]

* Improving Out-of-Distribution Generalization by Adversarial Training with Structured Priors (NeurIPS 2022) 
[[Paper](https://neurips.cc/virtual/2022/poster/53376)]

* Diverse Weight Averaging for Out-of-Distribution Generalization (NeurIPS 2022) 
[[Paper](https://neurips.cc/virtual/2022/poster/54194)]

* Using Mixup as a Regularizer Can Surprisingly Improve Accuracy & Out-of-Distribution Robustness (NeurIPS 2022) 
[[Paper](https://neurips.cc/virtual/2022/poster/53348)]

* Watermarking for Out-of-distribution Detection (NeurIPS 2022) 
[[Paper](https://neurips.cc/virtual/2022/poster/55165)]

* SIREN: Shaping Representations for Detecting Out-of-Distribution Objects (NeurIPS 2022) 
[[Paper](https://neurips.cc/virtual/2022/poster/54724)]

* Out-of-Distribution Detection via Conditional Kernel Independence Model (NeurIPS 2022) 
[[Paper](https://neurips.cc/virtual/2022/poster/54589)]

#### ICML
* Open-Sampling: Exploring Out-of-Distribution Data for Re-balancing long-tailed Datasets (ICML 2022) 
[[Paper](https://proceedings.mlr.press/v162/wei22c/wei22c.pdf)]
[[Code](https://github.com/hongxin001/logitnorm_ood)]<br>
*Datasets: long-tailed CIFAR10/100, CelebA-5, Places-LT*<br>
*Task: Image Classification*

* Partial and Asymmetric Contrastive Learning for Out-of-Distribution Detection in Long-Tailed Recognition (ICML 2022) 
[[Paper](https://arxiv.org/abs/2207.01160)]
[[Code](https://github.com/amazon-research/long-tailed-ood-detection)]<br>
*Datasets: CIFAR10-LT, CIFAR100-LT, and ImageNet-LT*<br>
*Task: Image Classification*

* Training OOD Detectors in their Natural Habitats (ICML 2022) 
[[Paper](https://arxiv.org/abs/2202.03299)]
[[Code](https://github.com/jkatzsam/woods_ood)]<br>
*Datasets: CIFAR10, CIFAR100 (ID), SVHN, Textures, Places365, LSUN-Crop, LSUN-Resize*<br>
*Task: Image Classification*

* Model Agnostic Sample Reweighting for Out-of-Distribution Learning (ICML 2022) 
[[Paper](https://icml.cc/virtual/2022/poster/17867)]

* Predicting Out-of-Distribution Error with the Projection Norm (ICML 2022) 
[[Paper](https://icml.cc/virtual/2022/poster/17655)]

* Fishr: Invariant Gradient Variances for Out-of-Distribution Generalization (ICML 2022) 
[[Paper](https://icml.cc/virtual/2022/poster/17213)]

* POEM: Out-of-Distribution Detection with Posterior Sampling (ICML 2022) 
[[Paper](https://icml.cc/virtual/2022/poster/16651)]

* Improved StyleGAN-v2 based Inversion for Out-of-Distribution Images (ICML 2022) 
[[Paper](https://icml.cc/virtual/2022/poster/17161)]

* Scaling Out-of-Distribution Detection for Real-World Settings (ICML 2022) 
[[Paper](https://icml.cc/virtual/2022/poster/16907)]

* Breaking Down Out-of-Distribution Detection: Many Methods Based on OOD Training Data Estimate a Combination of the Same Core Quantities (ICML 2022) 
[[Paper](https://icml.cc/virtual/2022/poster/17469)]

* Improving Out-of-Distribution Robustness via Selective Augmentation (ICML 2022) 
[[Paper](https://icml.cc/virtual/2022/poster/18011)]

* Out-of-Distribution Detection with Deep Nearest Neighbors (ICML 2022) 
[[Paper](https://icml.cc/virtual/2022/poster/16493)]

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

* Provable Guarantees for Understanding Out-of-distribution Detection (AAAI 2022) 
[[Paper](https://aaai-2022.virtualchair.net/poster_AAAI3913.html)]

* On the Impact of Spurious Correlation for Out-of-Distribution Detection (AAAI 2022) 
[[Paper](https://aaai-2022.virtualchair.net/poster_AAAI4486.html)]

* OoDHDR-Codec: Out-of-Distribution Generalization for HDR Image Compression (AAAI 2022) 
[[Paper](https://aaai-2022.virtualchair.net/poster_AAAI8610.html)]

* Zero-Shot Out-of-Distribution Detection Based on the Pre-Trained Model CLIP (AAAI 2022) 
[[Paper](https://aaai-2022.virtualchair.net/poster_AAAI10297.html)]

* iDECODe: In-Distribution Equivariance for Conformal Out-of-Distribution Detection (AAAI 2022) 
[[Paper](https://aaai-2022.virtualchair.net/poster_AAAI12912.html)]

* Exploiting Mixed Unlabeled Data for Detecting Samples of Seen and Unseen Out-of-Distribution Classes (AAAI 2022) 
[[Paper](https://aaai-2022.virtualchair.net/poster_AAAI6640.html)]

* VITA: A Multi-Source Vicinal Transfer Augmentation Method for Out-of-Distribution Generalization (AAAI 2022) 
[[Paper](https://aaai-2022.virtualchair.net/poster_AAAI733.html)]

* Learning Modular Structures That Generalize Out-of-Distribution (Student Abstract) (AAAI 2022) 
[[Paper](https://aaai-2022.virtualchair.net/poster_SA398.html)]


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

#### WACV
* Addressing Out-of-Distribution Label Noise in Webly-Labelled Data (WACV 2022) 
[[Paper](http://arxiv.org/abs/2110.13699)]
[[Code](https://git.io/JKGcj)]<br>
*Datasets: CIFAR-100, ImageNet32, MiniImageNet, Stanford Cars, mini-WebVision, ILSVRC12, Clothing1M*<br>
*Task: Image Classification*
<!--#### IJCV
#### BMVC
#### ICCw -->
#### BMVC 
* OSM: An Open Set Matting Framework with OOD Detection and Few-Shot Matting (BMVC 2022) 
[[Paper](https://bmvc2022.mpi-inf.mpg.de/0092.pdf)]<br>
*Datasets: SIMD*<br>
*Task: Out-of-Distribution Detection, Semantic Image Matting*

* VL4Pose: Active Learning Through Out-Of-Distribution Detection For Pose Estimation (BMVC 2022) 
[[Paper](https://bmvc2022.mpi-inf.mpg.de/610/)]
[[Code](https://github.com/meghshukla/ActiveLearningForHumanPose)]<br>
*Datasets: MPII, LSP/LSPET, ICVL*<br>
*Task: Active Learning*

* Shifting Transformation Learning for Robust Out-of-Distribution Detection (BMVC 2022) 
[[Paper](https://bmvc2022.mpi-inf.mpg.de/0679.pdf)]<br>
*Datasets: CIFAR-10, CIFAR-100, ImageNet-30*<br>
*Task: Image Classification*

<!----------------------------------------------------------------------------------------------------------------------------------------------->
### 2021 Papers
#### CVPR
* Deep Stable Learning for Out-of-Distribution Generalization (CVPR 2021) 
[[Paper](http://arxiv.org/abs/2104.07876)]

* MOOD: Multi-Level Out-of-Distribution Detection (CVPR 2021) 
[[Paper](http://arxiv.org/abs/2104.14726)]

* MOS: Towards Scaling Out-of-Distribution Detection for Large Semantic Space (CVPR 2021) 
[[Paper](http://arxiv.org/abs/2105.01879)]

* Out-of-Distribution Detection Using Union of 1-Dimensional Subspaces (CVPR 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zaeemzadeh_Out-of-Distribution_Detection_Using_Union_of_1-Dimensional_Subspaces_CVPR_2021_paper.pdf)]

#### ICLR
* Multiscale Score Matching for Out-of-Distribution Detection (ICLR 2021) 
[[Paper](https://openreview.net/forum?id=xoHdgbQJohv)]

* In-N-Out: Pre-Training and Self-Training using Auxiliary Information for Out-of-Distribution Robustness (ICLR 2021) 
[[Paper](https://openreview.net/forum?id=jznizqvr15J)]

* Understanding the failure modes of out-of-distribution generalization (ICLR 2021) 
[[Paper](https://openreview.net/forum?id=fSTD6NFIW_b)]

* Removing Undesirable Feature Contributions Using Out-of-Distribution Data (ICLR 2021) 
[[Paper](https://openreview.net/forum?id=eIHYL6fpbkA)]

#### NeurIPS
* Out-of-Distribution Generalization in Kernel Regression (NeurIPS 2021) 
[[Paper](https://neurips.cc/virtual/2021/poster/28011)]

* Invariance Principle Meets Information Bottleneck for Out-of-Distribution Generalization (NeurIPS 2021) 
[[Paper](https://neurips.cc/virtual/2021/poster/26541)]

* STEP: Out-of-Distribution Detection in the Presence of Limited In-Distribution Labeled Data (NeurIPS 2021) 
[[Paper](https://neurips.cc/virtual/2021/poster/28380)]

* The Out-of-Distribution Problem in Explainability and Search Methods for Feature Importance Explanations
 (NeurIPS 2021) 
[[Paper](https://neurips.cc/virtual/2021/poster/27019)]

* A Winning Hand: Compressing Deep Networks Can Improve Out-of-Distribution Robustness (NeurIPS 2021) 
[[Paper](https://neurips.cc/virtual/2021/poster/26319)]

* Locally Most Powerful Bayesian Test for Out-of-Distribution Detection using Deep Generative Models (NeurIPS 2021) 
[[Paper](https://neurips.cc/virtual/2021/poster/28170)]

* Task-Agnostic Undesirable Feature Deactivation Using Out-of-Distribution Data (NeurIPS 2021) 
[[Paper](https://neurips.cc/virtual/2021/poster/33051)]

* Exploring the Limits of Out-of-Distribution Detection (NeurIPS 2021) 
[[Paper](https://neurips.cc/virtual/2021/poster/27825)]

* ReAct: Out-of-distribution Detection With Rectified Activations (NeurIPS 2021) 
[[Paper](https://neurips.cc/virtual/2021/poster/27015)]

* Learning Causal Semantic Representation for Out-of-Distribution Prediction (NeurIPS 2021) 
[[Paper](https://neurips.cc/virtual/2021/poster/27172)]

* Towards a Theoretical Framework of Out-of-Distribution Generalization (NeurIPS 2021) 
[[Paper](https://neurips.cc/virtual/2021/poster/27952)]

* Characterizing Generalization under Out-Of-Distribution Shifts in Deep Metric Learning (NeurIPS 2021) 
[[Paper](https://neurips.cc/virtual/2021/poster/28292)]

* Single Layer Predictive Normalized Maximum Likelihood for Out-of-Distribution Detection (NeurIPS 2021) 
[[Paper](https://neurips.cc/virtual/2021/poster/28421)]

* On the Out-of-distribution Generalization of Probabilistic Image Modelling (NeurIPS 2021) 
[[Paper](https://neurips.cc/virtual/2021/poster/28832)]

* POODLE: Improving Few-shot Learning via Penalizing Out-of-Distribution Samples (NeurIPS 2021) 
[[Paper](https://neurips.cc/virtual/2021/poster/26921)]

* Towards optimally abstaining from prediction with OOD test examples (NeurIPS 2021) 
[[Paper](https://neurips.cc/virtual/2021/poster/28648)]

#### ICCV
* Trash To Treasure: Harvesting OOD Data With Cross-Modal Matching for Open-Set Semi-Supervised Learning (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Huang_Trash_To_Treasure_Harvesting_OOD_Data_With_Cross-Modal_Matching_for_ICCV_2021_paper.pdf)]<br>
*Datasets: CIFAR-10, Animal-10, Tiny-ImageNet, CIFAR100*<br>
*Task: Image Classification*

* Entropy Maximization and Meta Classification for Out-of-Distribution Detection in Semantic Segmentation (ICCV 2021) 
[[Paper](http://arxiv.org/abs/2012.06575)]

* The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution Generalization (ICCV 2021) 
[[Paper](http://arxiv.org/abs/2006.16241)]

* MG-GAN: A Multi-Generator Model Preventing Out-of-Distribution Samples in Pedestrian Trajectory Prediction (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Dendorfer_MG-GAN_A_Multi-Generator_Model_Preventing_Out-of-Distribution_Samples_in_Pedestrian_Trajectory_ICCV_2021_paper.pdf)]

* Semantically Coherent Out-of-Distribution Detection (ICCV 2021) 
[[Paper](http://arxiv.org/abs/2108.11941)]

* Linguistically Routing Capsule Network for Out-of-Distribution Visual Question Answering (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Cao_Linguistically_Routing_Capsule_Network_for_Out-of-Distribution_Visual_Question_Answering_ICCV_2021_paper.pdf)]

* CODEs: Chamfer Out-of-Distribution Examples Against Overconfidence Issue (ICCV 2021) 
[[Paper](http://arxiv.org/abs/2108.06024)]

* NAS-OoD: Neural Architecture Search for Out-of-Distribution Generalization (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Bai_NAS-OoD_Neural_Architecture_Search_for_Out-of-Distribution_Generalization_ICCV_2021_paper.pdf)]

* Triggering Failures: Out-of-Distribution Detection by Learning From Local Adversarial Attacks in Semantic Segmentation (ICCV 2021) 
[[Paper](http://arxiv.org/abs/2108.01634)]

#### ICML

* Understanding Failures in Out-of-Distribution Detection with Deep Generative Models (ICML 2021) 
[[Paper](https://icml.cc/virtual/2021/poster/9421)]

* Graph Convolution for Semi-Supervised Classification: Improved Linear Separability and Out-of-Distribution Generalization (ICML 2021) 
[[Paper](https://icml.cc/virtual/2021/poster/8741)]

* Out-of-Distribution Generalization via Risk Extrapolation (REx) (ICML 2021) 
[[Paper](https://icml.cc/virtual/2021/poster/9185)]

* Can Subnetwork Structure Be the Key to Out-of-Distribution Generalization? (ICML 2021) 
[[Paper](https://icml.cc/virtual/2021/poster/9481)]

* Accuracy on the Line: on the Strong Correlation Between Out-of-Distribution and In-Distribution Generalization (ICML 2021) 
[[Paper](https://icml.cc/virtual/2021/poster/9243)]

* Amortized Conditional Normalized Maximum Likelihood: Reliable Out of Distribution Uncertainty Estimation (ICML 2021) 
[[Paper](https://icml.cc/virtual/2021/poster/9305)]

* Improved OOD Generalization via Adversarial Training and Pretraing (ICML 2021) 
[[Paper](https://icml.cc/virtual/2021/poster/10511)]

<!--#### IEEE-Access
#### ECCV
#### AAAI
#### TPAMI-->
#### CVPRw
* Sample-Free White-Box Out-of-Distribution Detection for Deep Learning (CVPRw 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021W/TCV/papers/Begon_Sample-Free_White-Box_Out-of-Distribution_Detection_for_Deep_Learning_CVPRW_2021_paper.pdf)]

* Out-of-Distribution Detection and Generation Using Soft Brownian Offset Sampling and Autoencoders (CVPRw 2021) 
[[Paper](http://arxiv.org/abs/2105.02965)]

* DeVLBert: Out-of-Distribution Visio-Linguistic Pretraining With Causality (CVPRw 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021W/CiV/papers/Zhang_DeVLBert_Out-of-Distribution_Visio-Linguistic_Pretraining_With_Causality_CVPRW_2021_paper.pdf)]

<!--#### WACV
#### IJCV-->
#### BMVC

* OODformer: Out-Of-Distribution Detection Transformer (BMVC 2021) 
[[Paper](https://www.bmvc2021-virtualconference.com/assets/papers/1391.pdf)]
[[Code](https://github.com/rajatkoner08/oodformer)]<br>
*Datasets: CIFAR-10/-100 and ImageNet30*<br>
*Task: Image Classification*

#### ICCw

* SOoD: Self-Supervised Out-of-Distribution Detection Under Domain Shift for Multi-Class Colorectal Cancer Tissue Types (ICCVw 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021W/CVAMD/papers/Bozorgtabar_SOoD_Self-Supervised_Out-of-Distribution_Detection_Under_Domain_Shift_for_Multi-Class_Colorectal_ICCVW_2021_paper.pdf)]

#### Arxiv & Others

* ATOM: Robustifying Out-of-distribution Detection Using Outlier Mining (ECML 2021) 
[[Paper](https://arxiv.org/abs/2006.15207)] 
<!----------------------------------------------------------------------------------------------------------------------------------------------->
### 2020 Papers
#### CVPR

* Generalized ODIN: Detecting Out-of-Distribution Image Without Learning From Out-of-Distribution Data (CVPR 2020) 
[[Paper](http://arxiv.org/abs/2002.11297)]<br>
*Datasets: CIFAR10, CIFAR100, SVHN, TinyImageNet, LSUN and iSUN, DomainNet*<br>
*Task: Out-of-Distribution Image Classification*

#### ICLR

* Input Complexity and Out-of-distribution Detection with Likelihood-based Generative Models (ICLR 2020) 
[[Paper](https://openreview.net/forum?id=SyxIWpVYvr)]
[[Code](https://paperswithcode.com/paper/?openreview=SyxIWpVYvr)]<br>
*Datasets: CIFAR-10, CIFAR-100, CelebA, Fashion-MNIST, ImageNet, SVHN*<br>
*Task: Out-of-Distribution Image Classification*

* Learning to Balance: Bayesian Meta-Learning for Imbalanced and Out-of-distribution Tasks (ICLR 2020 Oral) 
[[Paper](https://openreview.net/forum?id=rkeZIJBYvr)]
[[Code](https://github.com/haebeom-lee/l2b)]<br>
*Datasets: CIFAR-10, CIFAR-100, CIFAR-FS, miniImageNet, SVHN, CUB, Aircraft, QuickDraw, and VGG-Flower, Traffic Signs, Fashion-MNIST*<br>
*Task: Out-of-Distribution Image Classification*

#### NeurIPS

* Why Normalizing Flows Fail to Detect Out-of-Distribution Data (NeurIPS 2020) 
[[Paper](https://neurips.cc/virtual/2020/poster/18088)]

* Certifiably Adversarially Robust Detection of Out-of-Distribution Data (NeurIPS 2020) 
[[Paper](https://neurips.cc/virtual/2020/poster/18869)]

* Likelihood Regret: An Out-of-Distribution Detection Score For Variational Auto-encoder (NeurIPS 2020) 
[[Paper](https://neurips.cc/virtual/2020/poster/16848)]

* Towards Maximizing the Representation Gap between In-Domain & Out-of-Distribution Examples (NeurIPS 2020) 
[[Paper](https://neurips.cc/virtual/2020/poster/17642)]

* Energy-based Out-of-distribution Detection (NeurIPS 2020) 
[[Paper](https://neurips.cc/virtual/2020/poster/17313)]

* On the Value of Out-of-Distribution Testing: An Example of Goodhart's Law (NeurIPS 2020) 
[[Paper](https://neurips.cc/virtual/2020/poster/16833)]

* OOD-MAML: Meta-Learning for Few-Shot Out-of-Distribution Detection and Classification (NeurIPS 2020) 
[[Paper](https://neurips.cc/virtual/2020/poster/17004)]

#### ICML

* Detecting Out-of-Distribution Examples with Gram Matrices (ICML 2020) 
[[Paper](https://arxiv.org/abs/1912.12510)]
[[Code](https://github.com/VectorInstitute/gram-ood-detection)]<br>
*Datasets: CIFAR10, CIFAR100, MNIST, SVHN, TinyImageNet, LSUN and iSUN*<br>
*Task: Out-of-Distribution Image Classification*

#### ECCV

* A Boundary Based Out-of-Distribution Classifier for Generalized Zero-Shot Learning (ECCV 2020) 
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690562.pdf)]<br>
*Datasets: AWA1, AWA2, CUB, FLO and SUN*<br>
*Task: Out-of-Distribution Image Classification*

#### CVPRw

* On Out-of-Distribution Detection Algorithms With Deep Neural Skin Cancer Classifiers (CVPRw 2020) 
[[Paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w42/Pacheco_On_Out-of-Distribution_Detection_Algorithms_With_Deep_Neural_Skin_Cancer_Classifiers_CVPRW_2020_paper.pdf)]

* Detection and Retrieval of Out-of-Distribution Objects in Semantic Segmentation (CVPRw 2020) 
[[Paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w20/Oberdiek_Detection_and_Retrieval_of_Out-of-Distribution_Objects_in_Semantic_Segmentation_CVPRW_2020_paper.pdf)]
<!----------------------------------------------------------------------------------------------------------------------------------------------->

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

* Out-of-Distribution Detection using Multiple Semantic Label Representations (NeurIPS 2018) 
[[Paper](https://papers.nips.cc/paper/2018/hash/2151b4c76b4dcb048d06a5c32942b6f6-Abstract.html)]
[[Code](http://www.github.com/MLSpeech/semantic_OOD)]<br>
*Datasets: CIFAR-10, CIFAR-100 and Google Speech Commands Dataset*<br>
*Task: Out-of-Distribution Image, Speech Classification*

* Likelihood Ratios for Out-of-Distribution Detection (NeurIPS 2019) 
[[Paper](https://papers.nips.cc/paper/2019/hash/1e79596878b2320cac26dd792a6c51c9-Abstract.html)]<br>
*Datasets: FashionMNIST -> MNIST, CIFAR10 -> SVHN, ImageNet and LSUN*<br>
*Task: Out-of-Distribution Image Classification*

* Out-of-Distribution Detection Using an Ensemble of Self Supervised Leave-out Classifiers (ECCV 2018) 
[[Paper](http://arxiv.org/abs/1809.03576)]<br>
*Datasets: CIFAR10, CIFAR100, TinyImageNet, LSUN, Uniform Noise(UNFM), Gaussian Noise(GSSN), iSUN*<br>
*Task: Out-of-Distribution Image Classification*

* Training Confidence-calibrated Classifiers for Detecting Out-of-Distribution Samples (ICLR 2018) 
[[Paper](https://openreview.net/forum?id=ryiAv2xAZ)]
[[Code](https://github.com/alinlab/Confident_classifier)]<br>
*Datasets: CIFAR-10, ImageNet, LSUN, MNIST, SVHN*<br>
*Task: Out-of-Distribution Image Classification*

* Unsupervised Out-of-Distribution Detection by Maximum Classifier Discrepancy (ICCV 2019) 
[[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Unsupervised_Out-of-Distribution_Detection_by_Maximum_Classifier_Discrepancy_ICCV_2019_paper.pdf)]<br>
*Datasets: CIFAR-10, CIFAR-100, LSUN, iSUN, TinyImageNet*<br>
*Task: Out-of-Distribution Image Classification*

* A Less Biased Evaluation of Out-of-distribution Sample Detectors (BMVC 2019) 
[[Paper](https://bmvc2019.org/wp-content/papers/0333.html)]
[[Code](https://github.com/ashafaei/OD-test)]<br>
*Datasets: CIFAR-10, CIFAR-100, MNIST, TinyImageNet, FashionMNIST, STL-10*<br>
*Task: Out-of-Distribution Image Classification*



<!----------------------------------------------------------------------------------------------------------------------------------------------->
## Open-Set Recognition 
<!----------------------------------------------------------------------------------------------------------------------------------------------->
### Surveys

* A Unified Survey on Anomaly, Novelty, Open-Set, and Out-of-Distribution Detection: Solutions and Future Challenges (TMLR 2023) 
[[Paper](https://arxiv.org/abs/2110.14051)]

* A Survey on Open Set Recognition (Arxiv 2021) 
[[Paper](https://arxiv.org/abs/2109.00893)]

* Toward Open-World Electroencephalogram Decoding Via Deep Learning: A Comprehensive Survey (IEEE Signal Processing Magazine 2021) 
[[Paper](https://arxiv.org/abs/2112.06654)]

* Recent advances in open set recognition: A survey (TPAMI 2020) 
[[Paper](https://arxiv.org/abs/1811.08581v4)]

* A Review of Open-World Learning and Steps Toward Open-World Learning Without Labels (Arxiv 2020) 
[[Paper](https://arxiv.org/abs/2011.12906)]

* A Survey of Open-World Person Re-Identification (IEEE Transactions on Circuits and Systems for Video Technology 2020) 
[[Paper](https://ieeexplore.ieee.org/document/8640834/)]

* Learning and the Unknown: Surveying Steps toward Open World Recognition (AAAI 2019) 
[[Paper](https://ojs.aaai.org/index.php/AAAI/article/download/5054/4927)]

* Transfer Adaptation Learning: A Decade Survey (Arxiv 2019) 
[[Paper](https://arxiv.org/abs/1903.04687)]

* Biometric-Enabled Authentication Machines: A Survey of Open-Set Real-World Applications (IEEE Transactions on Human-Machine Systems 2016) 
[[Paper](https://ieeexplore.ieee.org/document/7103330/)]

* Completeness, Recall, and Negation in Open-World Knowledge Bases: A Survey (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.05403)]


<!----------------------------------------------------------------------------------------------------------------------------------------------->
### 2023 Papers
#### CVPR
* Detecting Everything in the Open World: Towards Universal Object Detection (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.11749)]

* Annealing-based Label-Transfer Learning for Open World Object Detection (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Ma_Annealing-Based_Label-Transfer_Learning_for_Open_World_Object_Detection_CVPR_2023_paper.html)]

* Glocal Energy-based Learning for Few-Shot Open-Set Recognition (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Glocal_Energy-Based_Learning_for_Few-Shot_Open-Set_Recognition_CVPR_2023_paper.html)]

* OpenMix: Exploring Outlier Samples for Misclassification Detection (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.17093)]

* CapDet: Unifying Dense Captioning and Open-World Detection Pretraining (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.02489)]

* Open-set Fine-grained Retrieval via Prompting Vision-Language Evaluator (CVPR 2023) 
[[Paper](https://readpaper.com/paper/1608794558669368064)]

* Open-Set Representation Learning through Combinatorial Embedding (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2106.15278)]

* PROB: Probabilistic Objectness for Open World Object Detection (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2212.01424)]

* GLIGEN: Open-Set Grounded Text-to-Image Generation (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2301.07093)]

* Learning to Generate Text-grounded Mask for Open-world Semantic Segmentation from Only Image-Text Pairs (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2212.00785)]

* Open-Set Likelihood Maximization for Few-Shot Learning (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2301.08390)]

* Adjustment and Alignment for Unbiased Open Set Domain Adaptation (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Adjustment_and_Alignment_for_Unbiased_Open_Set_Domain_Adaptation_CVPR_2023_paper.html)]

* Coreset Sampling from Open-Set for Fine-Grained Self-Supervised Learning (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.11101)]

* Progressive Open Space Expansion for Open Set Model Attribution (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.06877)]

* Cascade Evidential Learning for Open-world Weakly-supervised Temporal Action Localization (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Cascade_Evidential_Learning_for_Open-World_Weakly-Supervised_Temporal_Action_Localization_CVPR_2023_paper.html)]

* Open-set Semantic Segmentation for Point Clouds via Adversarial Prototype Framework (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Open-Set_Semantic_Segmentation_for_Point_Clouds_via_Adversarial_Prototype_Framework_CVPR_2023_paper.html)]

* Towards Open-World Segmentation of Parts (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Pan_Towards_Open-World_Segmentation_of_Parts_CVPR_2023_paper.html)]

* Open-World Multi-Task Control Through Goal-Aware Representation Learning and Adaptive Horizon Prediction (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2301.10034)]

* CAT: LoCalization and IdentificAtion Cascade Detection Transformer for Open-World Object Detection (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2301.01970)]

* Unknown Sniffer for Object Detection: Don't Turn a Blind Eye to Unknown Objects (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.13769)]

* BundleSDF: Neural 6-DoF Tracking and 3D Reconstruction of Unknown Objects (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.14158)]

* Discriminating Known from Unknown Objects via Structure-Enhanced Recurrent Variational AutoEncoder (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Wu_Discriminating_Known_From_Unknown_Objects_via_Structure-Enhanced_Recurrent_Variational_AutoEncoder_CVPR_2023_paper.html)]

* Unsupervised Object Localization: Observing the Background to Discover Objects (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2212.07834)]

* CLIP2: Contrastive Language-Image-Point Pretraining from Real-World Point Cloud Data (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.12417)]

* Unknown Sniffer for Object Detection: Don't Turn a Blind Eye to Unknown Objects (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.13769)]

* Learning Attention as Disentangler for Compositional Zero-shot Learning (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.15111)]


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

<!--#### NeurIPS-->
#### ICCV
* Random Boxes Are Open-world Object Detectors (ICCV 2023) 
[[Paper](https://arxiv.org/abs/2307.08249)]

#### ICML
* Towards Unbiased Training in Federated Open-world Semi-supervised Learning (ICML 2023) 
[[Paper](https://arxiv.org/abs/2305.00771)]


<!--#### IEEE-Access
#### ECCV
#### AAAI-->
#### ACL

* Open Set Relation Extraction via Unknown-Aware Training (ACL 2023) 
[[Paper](https://arxiv.org/abs/2306.04950)]

* Domain Incremental Lifelong Learning in an Open World (ACL 2023) 
[[Paper](https://arxiv.org/abs/2305.06555)]

* Long-Tailed Question Answering in an Open World (ACL 2023) 
[[Paper](https://arxiv.org/abs/2305.06557)]

* Decoupling Pseudo Label Disambiguation and Representation Learning for Generalized Intent Discovery (ACL 2023) 
[[Paper](https://arxiv.org/abs/2305.17699)]

* Towards Open-World Product Attribute Mining: A Lightly-Supervised Approach (ACL 2023) 
[[Paper](https://arxiv.org/abs/2305.18350)]

#### TPAMI

* Orientational Distribution Learning with Hierarchical Spatial Attention for Open Set Recognition (TPAMI 2023) 
[[Paper](https://ieeexplore.ieee.org/document/9978641/)]

* Extended T: Learning with Mixed Closed-set and Open-set Noisy Labels (TPAMI 2023) 
[[Paper](https://arxiv.org/abs/2012.00932)]

* Handling Open-set Noise and Novel Target Recognition in Domain Adaptive Semantic Segmentation (TPAMI 2023) 
[[Paper](https://ieeexplore.ieee.org/document/10048580/)]

* From Instance to Metric Calibration: A Unified Framework for Open-World Few-Shot Learning (TPAMI 2023) 
[[Paper](https://ieeexplore.ieee.org/document/10041935/)]

#### CVPRw

* Impact of Pseudo Depth on Open World Object Segmentation with Minimal User Guidance (CVPRw 2023) 
[[Paper](https://arxiv.org/abs/2304.05716)]

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
#### WACVw
* Dealing With the Novelty in Open Worlds (WACVw 2023) 
[[Workshop](https://openaccess.thecvf.com/WACV2023_workshops/DNOW)]

* MetaMax: Improved Open-Set Deep Neural Networks via Weibull Calibration (WACVw 2023) 
[[Paper](http://arxiv.org/abs/2211.10872)]

#### Other Journals and Conferences

* OpenCon: Open-world Contrastive Learning (TMLR 2023) 
[[Paper](https://openreview.net/forum?id=2wWJxtpFer)]
[[Code](https://github.com/deeplearning-wisc/opencon)]<br>
*Datasets: CIFAR-10, CIFAR-100, Imagenet-100*<br>
*Task: Image Classification and Domain Adaptation*

* KRADA: Known-region-aware Domain Alignment for Open-set Domain Adaptation in Semantic Segmentation (TMLR 2023) 
[[Paper](https://arxiv.org/abs/2106.06237)]

* A Wholistic View of Continual Learning with Deep Neural Networks: Forgotten Lessons and the Bridge to Active and Open World Learning (Neural Networks 2023) 
[[Paper](https://arxiv.org/abs/2009.01797)]

* Spatial-Temporal Exclusive Capsule Network for Open Set Action Recognition (IEEE Transactions on Multimedia 2023) 
[[Paper](https://ieeexplore.ieee.org/document/10058554/)]

* Multi-resolution Fusion Convolutional Network for Open Set Human Activity Recognition (IEEE Internet of Things Journal 2023) 
[[Paper](https://ieeexplore.ieee.org/document/10040612/)]

* Open-Set Object Detection Using Classification-Free Object Proposal and Instance-Level Contrastive Learning (IEEE Robotics and Automation Letters 2023) 
[[Paper](https://ieeexplore.ieee.org/document/10035923/)]

* An Open-Set Modulation Recognition Scheme with Deep Representation Learning (IEEE Communications Letters 2023) 
[[Paper](https://ieeexplore.ieee.org/document/10034661/)]

* Neuro-Symbolic World Models for Adapting to Open World Novelty (AAMAS 2023) 
[[Paper](https://arxiv.org/abs/2301.06294)]

* Object-agnostic Affordance Categorization via Unsupervised Learning of Graph Embeddings (JAIR 2023) 
[[Paper](https://arxiv.org/abs/2304.05989)]

* Few-Shot Open-Set Learning for On-Device Customization of KeyWord Spotting Systems (INTERSPEECH 2023) 
[[Paper](https://arxiv.org/abs/2306.02161)]

* Open-world Semi-supervised Novel Class Discovery (IJCAI 2023) 
[[Paper](https://arxiv.org/abs/2305.13095)][[Code](https://github.com/LiuJMzzZ/OpenNCD)]

* DiffuseExpand: Expanding dataset for 2D medical image segmentation using diffusion models (IJCAIw 2023) 
[[Paper](https://arxiv.org/abs/2304.13416)]

* SalienDet: A Saliency-based Feature Enhancement Algorithm for Object Detection for Autonomous Driving (submitted to IEEE Transactions on Intelligent Vehicles 2023) 
[[Paper](https://arxiv.org/abs/2305.06940)]

#### Arxiv & Others 

* CAT: LoCalization and IdentificAtion Cascade Detection Transformer for Open-World Object Detection (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2301.01970)]


* ConceptFusion: Open-set Multimodal 3D Mapping (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2302.07241)]

* Contrastive Credibility Propagation for Reliable Semi-Supervised Learning (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2211.09929)]

* Open-Set Automatic Target Recognition (ICASSP 2023 Submission) 
[[Paper](https://arxiv.org/abs/2211.05883)]


* Characterizing Novelty in the Military Domain (Submitted to ICCRTS 2023) 
[[Paper](https://arxiv.org/abs/2302.12314)]

* Towards Open World NeRF-Based SLAM (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2301.03102)]

* Open-world Story Generation with Structured Knowledge Enhancement: A Comprehensive Survey (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2212.04634)]

* CoDEPS: Online Continual Learning for Depth Estimation and Panoptic Segmentation (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.10147)]

* Pixel-wise Gradient Uncertainty for Convolutional Neural Networks applied to Out-of-Distribution Segmentation (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.06920)]

* Finding Competence Regions in Domain Generalization (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.09989)]

* Background Matters: Enhancing Out-of-distribution Detection with Domain Features (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.08727)]

* A Simple Framework for Open-Vocabulary Segmentation and Detection (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.08131)]

* WDiscOOD: Out-of-Distribution Detection via Whitened Linear Discriminant Analysis (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.07543)]

* Towards Universal Vision-language Omni-supervised Segmentation (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.06547)]

* Open-world Instance Segmentation: Top-down Learning with Bottom-up Supervision (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.05503)]

* ElC-OIS: Ellipsoidal Clustering for Open-World Instance Segmentation on LiDAR Data (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.04351)]

* CLIP-FO3D: Learning Free Open-world 3D Scene Representations from 2D Dense CLIP (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.04748)]

* Distilled Reverse Attention Network for Open-world Compositional Zero-Shot Learning (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.00404)]

* Toward Defining a Domain Complexity Measure Across Domains (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.04141)]

* Methods and Mechanisms for Interactive Novelty Handling in Adversarial Environments (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2302.14208)]

* Detecting the open-world objects with the help of the Brain (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.11623)]

* Pluralistic Aging Diffusion Autoencoder (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.11086)]

* Open-World Pose Transfer via Sequential Test-Time Adaption (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.10945)]

* AUTO: Adaptive Outlier Optimization for Online Test-Time OOD Detection (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.12267)]

* Visually-Prompted Language Model for Fine-Grained Scene Graph Generation in an Open World (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.13233)]

* Addressing the Challenges of Open-World Object Detection (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.14930)]

* Troika: Multi-Path Cross-Modal Traction for Compositional Zero-Shot Learning (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.15230)]

* OpenInst: A Simple Query-Based Method for Open-World Instance Segmentation (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.15859)]

* RegionPLC: Regional Point-Language Contrastive Learning for Open-World 3D Scene Understanding (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2304.00962)]

* Establishing baselines and introducing TernaryMixOE for fine-grained out-of-distribution detection (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.17658)]

* Iterative Prompt Learning for Unsupervised Backlit Image Enhancement (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.17569)]

* Open Set Classification of GAN-based Image Manipulations via a ViT-based Hybrid Architecture (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2304.05212)]

* Uncertainty-inspired Open Set Learning for Retinal Anomaly Identification (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2304.03981)]

* Adaptive Negative Evidential Deep Learning for Open-set Semi-supervised Learning (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.12091)]

* Linking generative semi-supervised learning and generative open-set recognition (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.11702)]

* Self-Paced Learning for Open-Set Domain Adaptation (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.05933)]

* Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.05499)]

* Imbalanced Open Set Domain Adaptation via Moving-threshold Estimation and Gradual Alignment (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.04393)]

* Open Set Wireless Signal Classification: Augmenting Deep Learning with Expert Feature Classifiers (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2302.03749)]

* Improving Open-Set Semi-Supervised Learning with Self-Supervision (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2301.10127)]

* Hybrid Open-set Segmentation with Synthetic Negative Data (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2301.08555)]

* Subject-Independent Brain-Computer Interfaces with Open-Set Subject Recognition (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2301.07894)]

* Open-World Weakly-Supervised Object Localization (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2304.08271)]

* USNID: A Framework for Unsupervised and Semi-supervised New Intent Discovery (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2304.07699)]

* DoUnseen: Zero-Shot Object Detection for Robotic Grasping (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2304.02833)]

* Simple Domain Generalization Methods are Strong Baselines for Open Domain Generalization (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.18031)]

* CNS-Net: Conservative Novelty Synthesizing Network for Malware Recognition in an Open-set Scenario (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.01236)]

* MDENet: Multi-modal Dual-embedding Networks for Malware Open-set Recognition (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.01245)]

* Unlocking the Power of Open Set : A New Perspective for Open-set Noisy Label Learning (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.04203)]

* HiNoVa: A Novel Open-Set Detection Method for Automating RF Device Authentication (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.09594)]

* torchosr -- a PyTorch extension package for Open Set Recognition models evaluation in Python (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.09646)]

* Learning for Open-World Calibration with Graph Neural Networks (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.12039)]

* Instance-Level Semantic Maps for Vision Language Navigation (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.12363)]

* Online Open-set Semi-supervised Object Detection via Semi-supervised Outlier Filtering (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.13802)]

* Multi-modal Queried Object Detection in the Wild (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.18980)]

* In or Out? Fixing ImageNet Out-of-Distribution Detection Evaluation (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2306.00826)]

* Uncovering the Hidden Dynamics of Video Self-supervised Learning under Distribution Shifts (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2306.02014)]

* ProTeCt: Prompt Tuning for Hierarchical Consistency (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2306.02240)]


* 2D Supervised Monocular 3D Object Detection by Global-to-Local 3D Reconstruction (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2306.05418)]

* Open-World Continual Learning: Unifying Novelty Detection and Continual Learning (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2304.10038)]

* Uncovering CWE-CVE-CPE Relations with Threat Knowledge Graphs (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.00632)]

* Detecting Novelties with Empty Classes (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.00983)]

* ChinaOpen: A Dataset for Open-world Multimodal Learning (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.05880)]

* ZeroFlow: Fast Zero Label Scene Flow via Distillation (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.10424)]

* OpenShape: Scaling Up 3D Shape Representation Towards Open-World Understanding (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.10424)][[Code](https://colin97.github.io/OpenShape/)]

* MonoTDP: Twin Depth Perception for Monocular 3D Object Detection in Adverse Scenes (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.10974)]

* Learning for Open-World Calibration with Graph Neural Networks (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.12039)]

* WOT-Class: Weakly Supervised Open-world Text Classification (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.12401)]

* Task-Oriented Communication with Out-of-Distribution Detection: An Information Bottleneck Framework (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.12423)]

* GraphCare: Enhancing Healthcare Predictions with Open-World Personalized Knowledge Graphs (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.12788)]

* Matcher: Segment Anything with One Shot Using All-Purpose Feature Matching (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.13310)]

* Open-world Semi-supervised Generalized Relation Discovery Aligned in a Real-world Setting (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.13533)]

* PIVOINE: Instruction Tuning for Open-world Information Extraction (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.14898)]

* SPRING: GPT-4 Out-performs RL Algorithms by Studying Papers and Reasoning (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.15486)]

* Fairness Continual Learning Approach to Semantic Scene Understanding in Open-World Environments (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.15700)]

* Discovering Novel Actions in an Open World with Object-Grounded Visual Commonsense Reasoning (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.16602)]

* Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.17144)]

* VoxDet: Voxel Learning for Novel Instance Detection (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.17220)]

* Integrating Action Knowledge and LLMs for Task Planning and Situation Handling in Open Worlds (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.17590)]


* Multi-modal Queried Object Detection in the Wild (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.18980)]

* Temporal-controlled Frame Swap for Generating High-Fidelity Stereo Driving Data for Autonomy Analysis (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2306.01704)]

* Open-world Text-specified Object Counting (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2306.01851)]

* DOS: Diverse Outlier Sampling for Out-of-Distribution Detection (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2306.02031)]

* USD: Unknown Sensitive Detector Empowered by Decoupled Objectness and Segment Anything Model (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2306.02275)]



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

* Open-World Semi-Supervised Learning (ICLR 2022) 
[[Paper](https://openreview.net/forum?id=O-r8LOR-CCA)]
[[Code](https://github.com/snap-stanford/orca)]<br>
*Datasets: CIFAR-10, CIFAR-100, ImageNet, single-cell Mouse Ageing Cell Atlas dataset*<br>
*Task: Image Classification*

* Audio Lottery: Speech Recognition Made Ultra-Lightweight, Noise-Robust, and Transferable (ICLR 2022) 
[[Paper](https://openreview.net/forum?id=9Nk6AJkVYB)]
[[Code](https://github.com/VITA-Group/Audio-Lottery)]<br>
*Datasets: TED-LIUM, Common Voice, LibriSpeech*<br>
*Task: Lightweight Speech Recognition*

* Benchmarking the Spectrum of Agent Capabilities (ICLR 2022) 
[[Paper](https://openreview.net/forum?id=1W0z96MFEoH)]
[[Code](https://github.com/danijar/crafter)]<br>
*Datasets: Crafter*<br>
*Task: Agent Evaluation*

#### NeurIPS 
* Rethinking Knowledge Graph Evaluation Under the Open-World Assumption (NeurIPS 2022 Oral) 
[[Paper](https://openreview.net/forum?id=5xiLuNutzJG)]
[[Code](https://github.com/GraphPKU/Open-World-KG)]<br>
*Datasets: family tree KG*<br>
*Task: Knowledge  Graph  Completion*


* OpenAUC: Towards AUC-Oriented Open-Set Recognition	(NeurIPS 2022) 
[[Paper](https://arxiv.org/pdf/2210.13458v1.pdf)]<br>
*Datasets: MNIST1, SVHN2, CIFAR10, CIFAR+10, CIFAR+50, TinyImageNet, CUB*<br>
*Task: Image Classification*

* Unknown-Aware Domain Adversarial Learning for Open-Set Domain Adaptation	(NeurIPS 2022) 
[[Paper](https://arxiv.org/pdf/2206.07551.pdf)]
[[Code](https://github.com/JoonHo-Jang/UADAL)]<br>
*Datasets: Office-31, Office-Home, VisDA*<br>
*Task: Domain Adaptation*

* 3DOS: Towards Open Set 3D Learning: Benchmarking and Understanding Semantic Novelty Detection on Point Clouds	(NeurIPS 2022) 
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

* DetCLIP: Dictionary-Enriched Visual-Concept Paralleled Pre-training for Open-world Detection (NeurIPS 2022) 
[[Paper](https://arxiv.org/abs/2209.09407)]<br>
*Datasets: LVIS, YFCC100m*<br>
*Task: Object Detection*

* What is Where by Looking: Weakly-Supervised Open-World Phrase-Grounding without Text Inputs (NeurIPS 2022) 
[[Paper](https://arxiv.org/abs/2206.09358)]
[[Code](https://github.com/talshaharabany/what-is-where-by-looking)]<br>
*Datasets: CUB-200-2011, Stanford Car, Stanford dogs, Flickr30k, ReferIt, Visual Genome (VG), MSCOCO 2014*<br>
*Task: Object Localization and Captioning*

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

* OpenLDN: Learning to Discover Novel Classes for Open-World Semi-Supervised Learning	(ECCV 2022)
[[Paper](https://arxiv.org/abs/2207.02261)]<br>

* Open-world Semantic Segmentation for LIDAR Point Clouds (ECCV 2022) 
[[Paper](https://arxiv.org/abs/2207.01452)]

#### AAAI

* Learngene: From Open-World to Your Learning Task	(AAAI 2022)
[[Paper](https://arxiv.org/abs/2106.06788v3)]<br>
*Datasets:CIFAR100, ImageNet100*<br>
*Task: Meta Learning*

* PMAL: Open Set Recognition via Robust Prototype Mining (AAAI 2022)
[[Paper](https://arxiv.org/abs/2203.08569v1)]<br>
*Datasets: MNIST, SVHN, CIFAR10, CIFAR+10, CIFAR+50, TinyImageNet*<br>
*Task: Image Classification*

* Learning Network Architecture for Open-Set Recognition (AAAI 2022)
[[Paper](https://ojs.aaai.org/index.php/AAAI/article/download/20246/version/18543/20005)]<br>
*Datasets: MNIST, SVHN, CIFAR10, CIFAR+10, CIFAR+50*<br>
*Task: Image Classification*

* LUNA: Localizing Unfamiliarity Near Acquaintance for Open-Set Long-Tailed Recognition (AAAI 2022)
[[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/19887)]<br>
*Datasets: ImageNet-LT, Places-LT, Marine Species (MS)-LT*<br>
*Task: Image Classification*

#### ACL

* TegTok: Augmenting Text Generation via Task-specific and Open-world Knowledge (ACL 2022) 
[[Paper](https://arxiv.org/abs/2203.08517)] 


#### TPAMI

* Open Long-Tailed RecognitionIn A Dynamic World	(TPAMI 2022) 
[[Paper](https://arxiv.org/pdf/2208.08349v1.pdf)]
[[Code](https://liuziwei7.github.io/projects/LongTail.html)]<br>
*Datasets: CIFAR-10-LT,CIFAR-100-LT, and iNaturalist-18, Places-LT,  MS1M-LT, SUN-LT*<br>
*Task: Image Classification*

* Convolutional Prototype Network for Open Set Recognition (TPAMI 2022) 
[[Paper](https://ieeexplore.ieee.org/document/9296325/)]

* Class-Specific Semantic Reconstruction for Open Set Recognition (TPAMI 2022) 
[[Paper](https://ieeexplore.ieee.org/document/9864101/)]

* OpenGAN: Open-Set Recognition Via Open Data Generation (TPAMI 2022) 
[[Paper](https://ieeexplore.ieee.org/document/9799769/)]

* Open World Entity Segmentation (TPAMI 2022) 
[[Paper](https://ieeexplore.ieee.org/document/9976289/)]

* Bayesian Embeddings for Few-Shot Open World Recognition (TPAMI 2022) 
[[Paper](https://ieeexplore.ieee.org/document/9875990/)]

* Learning Graph Embeddings for Open World Compositional Zero-Shot Learning (TPAMI 2022) 
[[Paper](https://ieeexplore.ieee.org/document/9745371/)]

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

#### IJCV 

* Open-set Adversarial Defense with Clean-Adversarial Mutual Learning (IJCV 2022) 
[[Paper](https://arxiv.org/abs/2202.05953)]

#### BMVC

* Dual Decision Improves Open-Set Panoptic Segmentation (BMVC 2022) 
[[Paper](https://arxiv.org/pdf/2207.02504v3.pdf)]
[[Code](https://github.com/jd730/EOPSN.git)]<br>
*Datasets: MS-COCO 2017*<br>
*Task: Panoptic Segmentation*

<!-- 
#### ICCVw -->

#### ECCVw

* General Image Descriptors for Open World Image Retrieval using ViT CLIP (ECCVw 2022) 
[[Paper](https://arxiv.org/abs/2210.11141)]

#### Other Journals and Conferences

* Open Set Domain Adaptation By Novel Class Discovery (ICME 2022) 
[[Paper](https://arxiv.org/abs/2203.03329v1)]<br>
*Datasets: Office-31, Office-Home, DomainNet*<br>
*Task: Image Classification and Domain Adaptation*

* Discovering Implicit Classes Achieves Open Set Domain Adaptation (ICME 2022) 
[[Paper](https://ieeexplore.ieee.org/document/9859594)]

* Open-World Object Detection via Discriminative Class Prototype Learning (ICIP 2022) 
[[Paper](https://arxiv.org/abs/2302.11757)]

* Multi-Attribute Open Set Recognition (GCPR 2022) 
[[Paper](https://arxiv.org/abs/2208.06809)]

* Semantic Abstraction: Open-World 3D Scene Understanding from 2D Vision-Language Models (Conference on Robot Learning 2022) 
[[Paper](https://arxiv.org/abs/2207.11514)]

* Uncertainty for Identifying Open-Set Errors in Visual Object Detection (IROS/RAL 2022) 
[[Paper](https://arxiv.org/abs/2104.01328)]

* One-Shot Open-Set Skeleton-Based Action Recognition (IEEE International Conference on Humanoid Robots 2022) 
[[Paper](https://arxiv.org/abs/2209.04288)]

* 3DOS: Towards 3D Open Set Learning -- Benchmarking and Understanding Semantic Novelty Detection on Point Clouds (NeurIPS 2022 Datasets and Benchmarks) 
[[Paper](https://arxiv.org/abs/2207.11554)]

* Dummy Prototypical Networks for Few-Shot Open-Set Keyword Spotting (INTERSPEECH 2022) 
[[Paper](https://arxiv.org/abs/2206.13691)]

* Open-Set Recognition with Gradient-Based Representations (ICIP 2022) 
[[Paper](https://arxiv.org/abs/2206.08229)]

* Collective Decision of One-vs-Rest Networks for Open Set Recognition (TNNLS 2022) 
[[Paper](https://arxiv.org/abs/2103.10230)]

* Incremental Learning from Low-labelled Stream Data in Open-Set Video Face Recognition (Pattern Recognition 2022) 
[[Paper](https://arxiv.org/abs/2012.09571)]

* Unified Probabilistic Deep Continual Learning through Generative Replay and Open Set Recognition (Jounal of Imaging 2022) 
[[Paper](https://arxiv.org/abs/1905.12019)]

* Spectral-Spatial Latent Reconstruction for Open-Set Hyperspectral Image Classification (IEEE Transactions on Image Processing 2022) 
[[Paper](https://ieeexplore.ieee.org/document/9846880/)]

* Deep-Learning-Based Open Set Fault Diagnosis by Extreme Value Theory (IEEE Transactions on Industrial Informatics 2022) 
[[Paper](https://ieeexplore.ieee.org/document/9394793/)]

* Evaluating Uncertainty Calibration for Open-Set Recognition (ICRA 2022) 
[[Paper](https://arxiv.org/abs/2205.07160)]

* OpenStance: Real-world Zero-shot Stance Detection (CoNLL 2022) 
[[Paper](https://arxiv.org/abs/2210.14299)]

* Generalized Intent Discovery: Learning from Open World Dialogue System (COLING 2022) 
[[Paper](https://arxiv.org/abs/2209.06030)]

* Open World Learning Graph Convolution for Latency Estimation in Routing Networks (IJCNN 2022) 
[[Paper](https://arxiv.org/abs/2207.14643)]

* RAPid-Learn: A Framework for Learning to Recover for Handling Novelties in Open-World Environments (ICDL 2022) 
[[Paper](https://arxiv.org/abs/2206.12493)]

* SCIM: Simultaneous Clustering, Inference, and Mapping for Open-World Semantic Scene Understanding (ISRR 2022) 
[[Paper](https://arxiv.org/abs/2206.10670)]

* Exploring the Open World Using Incremental Extreme Value Machines (ICPR 2022) 
[[Paper](https://arxiv.org/abs/2205.14892)]

* Region-Aware Metric Learning for Open World Semantic Segmentation via Meta-Channel Aggregation (IJCAI 2022) 
[[Paper](https://arxiv.org/abs/2205.08083)]

* OA-Mine: Open-World Attribute Mining for E-Commerce Products with Weak Supervision (WWW 2022) 
[[Paper](https://arxiv.org/abs/2204.13874)] 

* Contrastive Learning for Cross-Domain Open World Recognition (IROS 2022) 
[[Paper](https://arxiv.org/abs/2203.09257)] 

* Towards Unsupervised Open World Semantic Segmentation (UAI 2022, PMLR) 
[[Paper](https://arxiv.org/abs/2201.01073)] 

* Open-Set Face Identification on Few-Shot Gallery by Fine-Tuning (ICPR 2022) 
[[Paper](https://arxiv.org/abs/2301.01922)]

#### Arxiv & Others 

* NovelCraft: A Dataset for Novelty Detection and Discovery in Open Worlds (Arxiv 2022 - TMLR 2023 Under Review) 
[[Paper](https://arxiv.org/abs/2206.11736)]

* Oracle Analysis of Representations for Deep Open Set Detection (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2209.11350)]

* From Known to Unknown: Quality-aware Self-improving Graph Neural Network for Open Set Social Event Detection (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2208.06973)]

* Semantic Abstraction: Open-World 3D Scene Understanding from 2D Vision-Language Models (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2207.11514)]

* Prompt-driven efficient Open-set Semi-supervised Learning (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2209.14205)]

* OOD Augmentation May Be at Odds with Open-Set Recognition (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2206.04242)]

* Knowledge Distillation Meets Open-Set Semi-Supervised Learning (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2205.06701)]

* Open World DETR: Transformer based Open World Object Detection (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2212.02969)]

* ProCC: Progressive Cross-primitive Compatibility for Open-World Compositional Zero-Shot Learning (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2211.12417)]

* PointCLIP V2: Adapting CLIP for Powerful 3D Open-world Learning (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2211.11682)]

* Simple Primitives with Feasibility- and Contextuality-Dependence for Open-World Compositional Zero-shot Learning (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2211.02895)]

* An Open-World Lottery Ticket for Out-of-Domain Intent Classification (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2210.07071)]

* Robot Task Planning and Situation Handling in Open Worlds (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2210.01287)]

* Self-Trained Proposal Networks for the Open World (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2208.11050)]

* UnCommonSense: Informative Negative Knowledge about Everyday Concepts (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2208.09292)]

* Single-Stage Open-world Instance Segmentation with Cross-task Consistency Regularization (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2208.09023)]

* Learning to Generalize with Object-centric Agents in the Open World Survival Game Crafter (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2208.03374)]

* Enhancing Automated Software Traceability by Transfer Learning from Open-World Data (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2207.01084)]

* Artificial Open World for Evaluating AGI: a Conceptual Design (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2206.01044)]

* On Leveraging Variational Graph Embeddings for Open World Compositional Zero-Shot Learning (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2204.11848)] 

* Extending One-Stage Detection with Open-World Proposals (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2201.02302)] 

* Revisiting Open World Object Detection (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2201.00471)]

* Accurate Open-set Recognition for Memory Workload (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2212.08817)]

* PointCaM: Cut-and-Mix for Open-Set Point Cloud Analysis (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2212.02011)]

* Spatial-Temporal Attention Network for Open-Set Fine-Grained Image Recognition (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2211.13940)]

* Open-Set Object Detection Using Classification-free Object Proposal and Instance-level Contrastive Learning with Appendix (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2211.11530)]

* PartCom: Part Composition Learning for 3D Open-Set Recognition (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2211.10880)]

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
*Datasets: Pascal VOC, MS-COCO*<br>
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

* Exemplar-Based Open-Set Panoptic Segmentation Network (CVPR 2021) 
[[Paper](http://arxiv.org/abs/2105.08336)]
[[Code](https://cv.snu.ac.kr/research/EOPSN)]<br>
*Datasets: COCO*<br>
*Task: Semantic Segmentation*

* Open World Compositional Zero-Shot Learning (CVPR 2021) 
[[Paper](https://arxiv.org/abs/2101.12609)]

<!-- #### ICLR-->

#### NeurIPS

* Towards Open-World Feature Extrapolation: An Inductive Graph Learning Approach (NeurIPS 2021) 
[[Paper](https://papers.nips.cc/paper/2021/hash/a1c5aff9679455a233086e26b72b9a06-Abstract.html)]
[[Code](https://github.com/qitianwu/FATE)]<br>
*Datasets: UCI Machine Learning Repository: Gene, Protein, Robot, Drive, Calls and Github, Avazu and Criteo*<br>
*Task: Graph Learning*

* OpenMatch: Open-Set Semi-supervised Learning with Open-set Consistency Regularization (NeurIPS 2021) 
[[Paper](https://papers.nips.cc/paper/2021/hash/da11e8cd1811acb79ccf0fd62cd58f86-Abstract.html)]
[[Code](https://github.com/VisionLearningGroup/OP_Match)]<br>
*Datasets: CIFAR10/100 and ImageNet*<br>
*Task: Image Classification*

* Improving Contrastive Learning on Imbalanced Data via Open-World Sampling (NeurIPS 2021) 
[[Paper](https://papers.nips.cc/paper/2021/hash/2f37d10131f2a483a8dd005b3d14b0d9-Abstract.html)]
[[Code](https://github.com/VITA-Group/MAK)]<br>
*Datasets: ImageNet-100-LT, ImageNet-900 and ImageNet-Places-mix*<br>
*Task: Image Classification*

* Open-set Label Noise Can Improve Robustness Against Inherent Label Noise (NeurIPS 2021) 
[[Paper](https://papers.nips.cc/paper/2021/hash/428fca9bc1921c25c5121f9da7815cde-Abstract.html)]
[[Code](https://github.com/hongxin001/ODNL)]<br>
*Datasets: CIFAR-10, CIFAR-100 and Clothing1M, 300K Random Images, CIFAR-5m*<br>
*Task: Image Classification with Noisy Labels*

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

* Towards Open-World Recommendation: An Inductive Model-based Collaborative Filtering Approach (ICML 2021) 
[[Paper](https://arxiv.org/abs/2007.04833)]

<!--#### IEEE-Access
#### AAAI-->
#### TPAMI
* Adversarial Reciprocal Points Learning for Open Set Recognition (TPAMI 2021) 
[[Paper](https://arxiv.org/abs/2103.00953)]

* Polyhedral Conic Classifiers for Computer Vision Applications and Open Set Recognition (TPAMI 2021) 
[[Paper](https://ieeexplore.ieee.org/document/8798888/)]

#### EMNLP

* Open World Classification with Adaptive Negative Samples (EMNLP 2021) 
[[Paper](https://arxiv.org/abs/2303.05581)]


#### CVPRw

* Addressing Visual Search in Open and Closed Set Settings (CVPRw 2021) 
[[Paper](http://arxiv.org/abs/2012.06509)]

* Contextual Transformer Networks for Visual Recognition (CVPRw 2021) 
[[Paper](https://arxiv.org/abs/2107.12292)]

#### WACV

* Class Anchor Clustering: A Loss for Distance-Based Open Set Recognition (WACV 2021) 
[[Paper](https://openaccess.thecvf.com/content/WACV2021/papers/Miller_Class_Anchor_Clustering_A_Loss_for_Distance-Based_Open_Set_Recognition_WACV_2021_paper.pdf)]<br>
*Datasets: MNIST, SVHN, CIFAR-10, CIFAR+10, CIFAR+50, TinyImageNet*<br>
*Task: Image Classification*

* Automatic Open-World Reliability Assessment (WACV 2021) 
[[Paper](http://arxiv.org/abs/2011.05506)]<br>
*Datasets: EfficientNet-B3 on ImageNet*<br>
*Task: Image Classification Reliability Assessment*

* Object Recognition With Continual Open Set Domain Adaptation for Home Robot (WACV 2021) 
[[Paper](https://openaccess.thecvf.com/content/WACV2021/papers/Kishida_Object_Recognition_With_Continual_Open_Set_Domain_Adaptation_for_Home_WACV_2021_paper.pdf)]<br>
*Datasets: RGB-D Object Dataset (ROD), Autonomous Robot Indoor Dataset (ARID), OpenLORIS-object Dataset, CORe50 Dataset, iCubWorld Transformations Dataset, COSDA-HR dataset*<br>
*Task: Home Robot Object Detection*

* EvidentialMix: Learning With Combined Open-Set and Closed-Set Noisy Labels (WACV 2021) 
[[Paper](http://arxiv.org/abs/2011.05704)]
[[Code](https://github.com/ragavsachdeva/EvidentialMix)]<br>
*Datasets: CIFAR-10, CIFAR-100, ImageNet32*<br>
*Task: Image Classification*

<!--#### IJCV
#### BMVC
#### ICCw-->
#### NeurIPSw

* Open-Set Multi-Source Multi-Target Domain Adaptation (NeurIPSw 2021) 
[[Paper](https://arxiv.org/abs/2302.00995)]

#### Other Journals and Conferences

* Open-set Intersection Intention Prediction for Autonomous Driving (ICRA 2021) 
[[Paper](https://arxiv.org/abs/2103.00140)]

* Open-Set Support Vector Machines (TSMC 2021) 
[[Paper](https://arxiv.org/abs/1606.03802)]

* Empowering Knowledge Distillation via Open Set Recognition for Robust 3D Point Cloud Classification (Pattern Recognition Letters 2021) 
[[Paper](https://arxiv.org/abs/2010.13114)]

* Fully Convolutional Open Set Segmentation (Machine Learning 2021) 
[[Paper](https://link.springer.com/article/10.1007/s10994-021-06027-1)]

* O2D2: Out-Of-Distribution Detector to Capture Undecidable Trials in Authorship Verification (PAN@CLEF 2021) 
[[Paper](https://arxiv.org/abs/2106.15825)]

* Open Set Authorship Attribution toward Demystifying Victorian Periodicals (ICDAR 2021) 
[[Paper](https://arxiv.org/abs/1912.08259)]

* Siamese Network-based Open Set Identification of Communications Emitters with Comprehensive Features (CCISP 2021) 
[[Paper](https://ieeexplore.ieee.org/document/9639257/)]

* Recurrent Variational Open-Set Recognition (URTC 2021) 
[[Paper](https://ieeexplore.ieee.org/document/9701621/)]

* Unsupervised Domain Alignment Based Open Set Structural Recognition of Macromolecules Captured By Cryo-Electron Tomography (ICIP 2021) 
[[Paper](https://ieeexplore.ieee.org/document/9506205/)]

* The Open Set Weighted Domain Adaptation Method for Fault Diagnosis of Rolling Bearings (PHM-Nanjing 2021) 
[[Paper](https://ieeexplore.ieee.org/document/9612979/)]

* Open Set Domain Adaptation: Theoretical Bound and Algorithm (IEEE Transactions on Neural Networks and Learning Systems 2021) 
[[Paper](https://ieeexplore.ieee.org/document/9186366/)]

* Open-Set Human Identification Based on Gait Radar Micro-Doppler Signatures (IEEE Sensors Journal 2021) 
[[Paper](https://ieeexplore.ieee.org/document/9328322/)]

* On the Challenges of Open World Recognition under Shifting Visual Domains (RAL/ICRA 2021) 
[[Paper](https://arxiv.org/abs/2107.04461)] 

* Rapid Exploration for Open-World Navigation with Latent Goal Models (CoRL 2021) 
[[Paper](https://arxiv.org/abs/2104.05859)] 

* Lifelong Learning of Graph Neural Networks for Open-World Node Classification (IJCNN 2021) 
[[Paper](https://arxiv.org/abs/2006.14422)] 

* Towards Entity Alignment in the Open World: An Unsupervised Approach (DASFAA 2021) 
[[Paper](https://arxiv.org/abs/2101.10535)] 

#### Arxiv & Others

* Open-Set Representation Learning through Combinatorial Embedding (Arxiv 2021) 
[[Paper](https://arxiv.org/abs/2106.15278)]

* A Unified Benchmark for the Unknown Detection Capability of Deep Neural Networks (Arxiv 2021) 
[[Paper](https://arxiv.org/abs/2112.00337)]

* One-Class Meta-Learning: Towards Generalizable Few-Shot Open-Set Classification (Arxiv 2021) 
[[Paper](https://arxiv.org/abs/2109.06859)]

* Learning Metrics from Mean Teacher: A Supervised Learning Method for Improving the Generalization of Speaker Verification System (Arxiv 2021) 
[[Paper](https://arxiv.org/abs/2104.06604)]

* An Empirical Study and Analysis on Open-Set Semi-Supervised Learning (Arxiv 2021) 
[[Paper](https://arxiv.org/abs/2101.08237)]

* Learning Open-World Object Proposals without Learning to Classify (Arxiv 2021) 
[[Paper](https://arxiv.org/abs/2108.06753)] 

* Open-World Entity Segmentation (Arxiv 2021) 
[[Paper](https://arxiv.org/abs/2107.14228)] 

* Integrating Planning, Execution and Monitoring in the presence of Open World Novelties: Case Study of an Open World Monopoly Solver (Arxiv 2021) 
[[Paper](https://arxiv.org/abs/2107.04303)] 

* The Difficulty of Novelty Detection in Open-World Physical Domains: An Application to Angry Birds (Arxiv 2021) 
[[Paper](https://arxiv.org/abs/2106.08670)] 

* Open-world Machine Learning: Applications, Challenges, and Opportunities (Arxiv 2021) 
[[Paper](https://arxiv.org/abs/2105.13448)] 

* Towards Open-World Text-Guided Face Image Generation and Manipulation (Arxiv 2021) 
[[Paper](https://arxiv.org/abs/2104.08910)] 

* Learning without Seeing nor Knowing: Towards Open Zero-Shot Learning (Arxiv 2021) 
[[Paper](https://arxiv.org/abs/2103.12437)] 

* Out-distribution aware Self-training in an Open World Setting (Arxiv 2021) 
[[Paper](https://arxiv.org/abs/2012.12372)] 

* Self-Supervised Features Improve Open-World Learning (Arxiv 2021) 
[[Paper](https://arxiv.org/abs/2102.07848)] 

* Semantic Task Planning for Service Robots in Open World (Arxiv 2021) 
[[Paper](https://arxiv.org/abs/2011.00621)] 

<!----------------------------------------------------------------------------------------------------------------------------------------------->
### 2020 Papers
#### CVPR

* Few-Shot Open-Set Recognition Using Meta-Learning (CVPR 2020) 
[[Paper](http://arxiv.org/abs/2005.13713)]<br>
*Datasets: mini-Imagenet, CIFAR10, XJTU-Stevens*<br>
*Task: Image Classification*

* Towards Inheritable Models for Open-Set Domain Adaptation (CVPR 2020) 
[[Paper](http://arxiv.org/abs/2004.04388)]
[[Code](https://github.com/val-iisc/inheritune)]<br>
*Datasets: Office-31, Office-Home, VisDA*<br>
*Task: Domain Adaptation*

* Exploring Category-Agnostic Clusters for Open-Set Domain Adaptation (CVPR 2020) 
[[Paper](http://arxiv.org/abs/2006.06567)]<br>
*Datasets: Office, VisDA*<br>
*Task: Domain Adaptation*

* Conditional Gaussian Distribution Learning for Open Set Recognition (CVPR 2020) 
[[Paper](http://arxiv.org/abs/2003.08823)]<br>
*Datasets: MNIST, SVHN, CIFAR-10, CIFAR-100, Tiny-ImageNet, ImageNet-crop, ImageNet-resize,LSUN-crop, and LSUN-resize*<br>
*Task: Image Classification*

* Generative-Discriminative Feature Representations for Open-Set Recognition (CVPR 2020) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Perera_Generative-Discriminative_Feature_Representations_for_Open-Set_Recognition_CVPR_2020_paper.pdf)]<br>
*Datasets: SVHN, CIFAR-10, CIFAR+10, CIFAR+50, Tiny-ImageNet, ImageNet-crop, ImageNet-resize,LSUN-crop, and LSUN-resize*<br>
*Task: Image Classification*

#### ICML

* Progressive Graph Learning for Open-Set Domain Adaptation (ICML 2020) 
[[Paper](https://arxiv.org/abs/2006.12087)]<br>
*Datasets: Office-Home, VisDA-17, Syn2Real-O*<br>
*Task: Domain Adaptation*


#### ECCV

* Hybrid Models for Open Set Recognition (ECCV 2020) 
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480103.pdf)]<br>
*Datasets: MNIST, SVHN, CIFAR10, CIFAR100, CIFAR+10, CIFAR+50 and TinyImageNet*<br>
*Task: Image Classification*

* Learning Open Set Network with Discriminative Reciprocal Points (ECCV 2020) 
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480511.pdf)]<br>
*Datasets: MNIST, SVHN, CIFAR10, CIFAR100, CIFAR+10, CIFAR+50 and TinyImageNet, ImageNet-LT, Aircraft 300*<br>
*Task: Image Classification*

* Multi-Task Curriculum Framework for Open-Set Semi-Supervised Learning (ECCV 2020) 
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570426.pdf)]<br>
*Datasets: LSUN, TinyImageNet, Gaussian, Uniform, CIFAR-10, SVHN*<br>
*Task: Multi-task Learning*

* On the Effectiveness of Image Rotation for Open Set Domain Adaptation (ECCV 2020) 
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610409.pdf)]
[[Code](https://github.com/silvia1993/ROS)]<br>
*Datasets: Office-31 and Office-Home*<br>
*Task: Domain Adaptation*

* Open-set Adversarial Defense (ECCV 2020) 
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620664.pdf)]
[[Code](https://github.com/rshaojimmy/ECCV2020-OSAD)]<br>
*Datasets: SVHN, CIFAR10, TinyImageNet*<br>
*Task: Adversarial Defense*

* Multi-Source Open-Set Deep Adversarial Domain Adaptation (ECCV 2020) 
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710732.pdf)]<br>
*Datasets: Office-31 and Office-Home, Office-CalTech, Digits*<br>
*Task: Domain Adaptation*

* Representative-Discriminative Learning for Open-set Land Cover Classification of Satellite Imagery (ECCV 2020) 
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750001.pdf)]
[[Code](https://github.com/raziehkaviani/rdosr)]<br>
*Datasets: Pavia University (PU) and Pavia Center (PC), Indian Pines (IN), CIFAR10, TinyImageNet*<br>
*Task: Hyperspectral Image Classification*

* Learning to Detect Open Classes for Universal Domain Adaptation (ECCV 2020) 
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600562.pdf)]<br>
*Datasets: Office-31 and Office-Home, VisDA, DomainNet*<br>
*Task: Domain Adaptation*

* Adaptive Object Detection with Dual Multi-Label Prediction (ECCV 2020) 
[[Paper](https://arxiv.org/abs/2003.12943)]<br>


<!--#### IEEE-Access
#### AAAI-->
#### TPAMI

* Vocabulary-Informed Zero-Shot and Open-Set Learning (TPAMI 2020) 
[[Paper](https://ieeexplore.ieee.org/document/8734786/)]

* Unsupervised Multi-Class Domain Adaptation: Theory, Algorithms, and Practice (TPAMI 2020) 
[[Paper](https://arxiv.org/abs/2002.08681)]
[[Code](https://github.com/YBZh/MultiClassDA)]<br>

#### WACV

* The Overlooked Elephant of Object Detection: Open Set (WACV 2020) 
[[Paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Dhamija_The_Overlooked_Elephant_of_Object_Detection_Open_Set_WACV_2020_paper.pdf)]
[[Code](https://github.com/Vastlab/Elephant-of-object-detection)]<br>
*Datasets: PASCAL VOC, MSCOCO*<br>
*Task: Object Detection*

#### ICMLw

* An Open-World Simulated Environment for Developmental Robotics (ICMLw 2020) 
[[Paper](https://arxiv.org/abs/2007.09300)] 

* A Critical Evaluation of Open-World Machine Learning (ICMLw 2020) 
[[Paper](https://arxiv.org/abs/2007.04391)] 


#### ECCVw

* Improved Robustness to Open Set Inputs via Tempered Mixup (ECCVw 2020) 
[[Paper](https://arxiv.org/abs/2009.04659)]

#### Other Journals and Conferences

* Boosting Deep Open World Recognition by Clustering (IROS/RAL 2020) 
[[Paper](https://arxiv.org/abs/2004.13849)]

* Are Out-of-Distribution Detection Methods Effective on Large-Scale Datasets? (Plos One 2020) 
[[Paper](https://arxiv.org/abs/1910.14034)]

* Open-set Face Recognition for Small Galleries Using Siamese Networks (IWSSIP 2020) 
[[Paper](https://arxiv.org/abs/2105.06967)]

* Adversarial Network with Multiple Classifiers for Open Set Domain Adaptation (TMM 2020) 
[[Paper](https://arxiv.org/abs/2007.00384)]
[[Code](https://github.com/tasfia/DAMC)]<br>

* P-ODN: Prototype based Open Deep Network for Open Set Recognition (Scientific Reports 2020) 
[[Paper](https://arxiv.org/abs/1905.01851)]

* Towards Open-Set Semantic Segmentation of Aerial Images (LAGIRS 2020) 
[[Paper](https://arxiv.org/abs/2001.10063)]

* Centralized Large Margin Cosine Loss for Open-Set Deep Palmprint Recognition (IEEE Transactions on Circuits and Systems for Video Technology 2020) 
[[Paper](https://ieeexplore.ieee.org/document/8666165/)]

* Joint Partial Optimal Transport for Open Set Domain Adaptation (IJCAI 2020) 
[[Paper](https://www.ijcai.org/Proceedings/2020/0352.pdf)]

* Open-World Class Discovery with Kernel Networks (ICDM 2020 Best paper candidate) 
[[Paper](https://arxiv.org/abs/2012.06957)] 


#### Arxiv & Others

* One-vs-Rest Network-based Deep Probability Model for Open Set Recognition (Arxiv 2020) 
[[Paper](https://arxiv.org/abs/2004.08067)]

* Collaborative Training of Balanced Random Forests for Open Set Domain Adaptation (Arxiv 2020) 
[[Paper](https://arxiv.org/abs/2002.03642v1)]

* Known-class Aware Self-ensemble for Open Set Domain Adaptation (Arxiv 2020) 
[[Paper](https://arxiv.org/abs/1905.01068v1)]

* Probabilistic Case-based Reasoning for Open-World Knowledge Graph Completion (Arxiv 2020) 
[[Paper](https://arxiv.org/abs/2010.03548)] 


<!----------------------------------------------------------------------------------------------------------------------------------------------->
### 2019 Papers
#### CVPR
* C2AE: Class Conditioned Auto-Encoder for Open-set Recognition (CVPR 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Oza_C2AE_Class_Conditioned_Auto-Encoder_for_Open-Set_Recognition_CVPR_2019_paper.pdf)]
[[Code](https://github.com/otkupjnoz/c2ae)]<br>
*Datasets: MNIST, SVHN, CIFAR10, CIFAR+10, CIFAR+50, TinyImageNet*<br>
*Task: Image Classification*

* Large-Scale Long-Tailed Recognition in an Open World (CVPR 2019 Oral) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Large-Scale_Long-Tailed_Recognition_in_an_Open_World_CVPR_2019_paper.pdf)]
[[Code](https://liuziwei7.github.io/projects/LongTail.html)]<br>
*Datasets: ImageNet-LT (object-centric), Places-LT (scene-centric),and MS1M-LT (face-centric)*<br>
*Task: Domain Adaptation*

* Separate to Adapt: Open Set Domain Adaptation via Progressive Separation (CVPR 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Separate_to_Adapt_Open_Set_Domain_Adaptation_via_Progressive_Separation_CVPR_2019_paper.pdf)]
[[Code](github.com/thuml)]<br>
*Datasets: Office-31, Office-Home, VisDA-17, Digits, Caltech-ImageNet*<br>
*Task: Domain Adaptation*

* Classification-Reconstruction Learning for Open-Set Recognition (CVPR 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yoshihashi_Classification-Reconstruction_Learning_for_Open-Set_Recognition_CVPR_2019_paper.pdf)]
[[Code](https://github.com/facebookresearch/odin)]<br>
*Datasets: MNIST, CIFAR-10, SVHN, Tiny-ImageNet, and DBpedia*<br>
*Task: Image Classification-Reconstruction Learning*

* Weakly Supervised Open-Set Domain Adaptation by Dual-Domain Collaboration (CVPR 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Tan_Weakly_Supervised_Open-Set_Domain_Adaptation_by_Dual-Domain_Collaboration_CVPR_2019_paper.pdf)]<br>
*Datasets: DukeMTMC-reID, Office*<br>
*Task: Domain Adaptation*

* FineGAN: Unsupervised Hierarchical Disentanglement for Fine-Grained Object Generation and Discovery (CVPR 2019 Oral) 
[[Paper](https://arxiv.org/abs/1811.11155)]<br>


#### ICLR
* Learning Factorized Representations for Open-set Domain Adaptation (ICLR 2019) 
[[Paper](https://openreview.net/forum?id=SJe3HiC5KX)]<br>
*Datasets: Bing(B), Caltech256(C), ImageNet(I) and SUN(S), hence referred to as BCIS, Office, namely Amazon(A), DSLR(D) and Webcam(W)*<br>
*Task: Image Classification*

#### ICCV

* Attract or Distract: Exploit the Margin of Open Set (ICCV 2019) 
[[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Feng_Attract_or_Distract_Exploit_the_Margin_of_Open_Set_ICCV_2019_paper.pdf)]
[[Code](https://github.com/qy-feng/margin-openset.git)]<br>
*Datasets: Office-31, Digit*<br>
*Task: Domain Adaptation*
#### AAAI

* An Open-World Extension to Knowledge Graph Completion Models (AAAI 2019) 
[[Paper](https://arxiv.org/abs/1906.08382)] 

#### BMVC
* Generalized Zero-shot Learning using Open Set Recognition (BMVC 2019) 
[[Paper](https://bmvc2019.org/wp-content/papers/0035.html)]<br>
*Datasets: AWA1, APY, FLO, and CUB*<br>
*Task: Image Classification*

* Open-set Recognition of Unseen Macromolecules in Cellular Electron Cryo-Tomograms by Soft Large Margin Centralized Cosine Loss (BMVC 2019) 
[[Paper](https://bmvc2019.org/wp-content/papers/0347.html)]<br>
*Datasets: CECT, Simulation using PDB2VOL program*<br>
*Task: Subtomogram Recognition*

#### ICCVw

* Open Set Recognition Through Deep Neural Network Uncertainty: Does Out-of-Distribution Detection Require Generative Classifiers? (ICCVw 2019) 
[[Paper](https://openaccess.thecvf.com/content_ICCVW_2019/papers/SDL-CV/Mundt_Open_Set_Recognition_Through_Deep_Neural_Network_Uncertainty_Does_Out-of-Distribution_ICCVW_2019_paper.pdf)]

#### Arxiv & Others
* Open-world Learning and Application to Product Classification (WWW 2019) 
[[Paper](https://arxiv.org/pdf/1809.06004v2.pdf)]
[[Code](https://www.cs.uic.edu/~hxu/)]<br>
*Datasets:  product descriptions from the Amazon Datasets*<br>
*Task: Image Classification*

* The Importance of Metric Learning for Robotic Vision: Open Set Recognition and Active Learning (ICRA 2019) 
[[Paper](https://arxiv.org/abs/1902.10363)]

* Knowledge is Never Enough: Towards Web Aided Deep Open World Recognition (ICRA 2019) 
[[Paper](https://arxiv.org/abs/1906.01258)]

* Identifying Unknown Instances for Autonomous Driving (CoRL 2019) 
[[Paper](https://arxiv.org/abs/1910.11296)]

* Known-class Aware Self-ensemble for Open Set Domain Adaptation (Arxiv 2019) 
[[Paper](https://arxiv.org/abs/1905.01068)]

* An In-Depth Study on Open-Set Camera Model Identification (IEEE Access 2019) 
[[Paper](https://ieeexplore.ieee.org/document/8732341)]

* Open Set Incremental Learning for Automatic Target Recognition (IEEE Transactions on Geoscience and Remote Sensing 2019) 
[[Paper](https://ieeexplore.ieee.org/document/8631004/)]

* Multi-stage Deep Classifier Cascades for Open World Recognition (CIKM 2019) 
[[Paper](https://arxiv.org/abs/1908.09931)] 

* Enabling Open-World Specification Mining via Unsupervised Learning (Arxiv 2019) 
[[Paper](https://arxiv.org/abs/1904.12098)] 

* Open-world Learning and Application to Product Classification (WWW 2019) 
[[Paper](https://arxiv.org/abs/1809.06004)] 

<!----------------------------------------------------------------------------------------------------------------------------------------------->
### 2018 Papers

#### CVPR

* Iterative Learning With Open-Set Noisy Labels (CVPR 2018) 
[[Paper](http://arxiv.org/abs/1804.00092v1)]<br>
*Datasets: CIFAR-10+SVHN, CIFAR-10+CIFAR-100 and CIFAR-10+ImageNet3, CIFAR-100/ImageNet32*<br>
*Task: Image Classification*

* Towards Open-Set Identity Preserving Face Synthesis (CVPR 2018) 
[[Paper](http://arxiv.org/abs/1803.11182v2)]<br>
*Datasets: MS-Celeb-1M, LFW, Multi-PIE*<br>
*Task: Object Detection*

* Polarimetric Synthetic-Aperture-Radar Change-Type Classification With a Hyperparameter-Free Open-Set Classifier (CVPRw 2018) 
[[Paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w21/Koch_Polarimetric_Synthetic-Aperture-Radar_Change-Type_CVPR_2018_paper.pdf)]<br>

#### ICML

* Open Category Detection with PAC Guarantees (ICML 2018) 
[[Paper](http://proceedings.mlr.press/v80/liu18e.html)]<br>
*Datasets: Landsat, Opt.digits, pageb, Shuttle, Covertype and MNIST, Tiny ImageNet*<br>
*Task: Image Classification*

#### ECCV

* Bayesian Semantic Instance Segmentation in Open Set World (ECCV 2018) 
[[Paper](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Trung_Pham_Bayesian_Instance_Segmentation_ECCV_2018_paper.pdf)]<br>
*Datasets: COCO, NYU*<br>
*Task: Semantic Segmentation*

* Open Set Learning with Counterfactual Images (ECCV 2018) 
[[Paper](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Lawrence_Neal_Open_Set_Learning_ECCV_2018_paper.pdf)]<br>
*Datasets: MNIST, SVHN, CIFAR-10, and Tiny-Imagenet*<br>
*Task: Counterfactual Image Generation, Image Classification*

* Open Set Domain Adaptation by Backpropagation (ECCV 2018) 
[[Paper](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Kuniaki_Saito_Adversarial_Open_Set_ECCV_2018_paper.pdf)]<br>
*Datasets: Office, VisDA and digits*<br>
*Task: Domain Adaptation*

* Improving Generalization via Scalable Neighborhood Component Analysis (ECCV 2018) 
[[Paper](https://arxiv.org/abs/1808.04699)]<br>

#### AAAI

* Open-World Knowledge Graph Completion (AAAI 2018) 
[[Paper](https://arxiv.org/abs/1711.03438)] 

#### TPAMI
* Open Set Domain Adaptation for Image and Action Recognition (TPAMI 2018) 
[[Paper](https://arxiv.org/abs/1907.12865)]

* The Extreme Value Machine (TPAMI 2018) 
[[Paper](https://ieeexplore.ieee.org/document/7932895/)]


#### Arxiv & Others

* Dropout Sampling for Robust Object Detection in Open-Set Conditions (ICRA 2018) 
[[Paper](https://arxiv.org/abs/1710.06677)]


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

* Toward an Efficient Multi-class Classification in an Open Universe (Arxiv 2015) 
[[Paper](https://arxiv.org/abs/1511.00725)]
* Online Open World Recognition (Arxiv 2016) 
[[Paper](https://arxiv.org/abs/1604.02275)]<br>

* Towards Open World Recognition (CVPR 2015) 
[[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Bendale_Towards_Open_World_2015_CVPR_paper.pdf)]
[[Code](http://vast.uccs.edu/OpenWorld)]<br>
*Datasets: ImageNet 2010*<br>
*Task: Image Classification*

* Towards Open Set Deep Networks (OpenMax) (CVPR 2016) 
[[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Bendale_Towards_Open_Set_CVPR_2016_paper.pdf)]<br>
*Datasets: ILSVRC 2012*<br>
*Task: Image Classification*

* Generative OpenMax for multi-class open set classification (BMVC 2017) 
[[Paper](http://www.bmva.org/bmvc/2017/papers/paper042/paper042.pdf)]<br>
*Datasets: : MNIST, HASYv2*<br>
*Task: Image Classification*

* Probability Models for Open Set Recognition (TPAMI 2014) 
[[Paper](https://ieeexplore.ieee.org/document/6809169/)]

* Open Set Domain Adaptation (ICCV 2017) 
[[Paper](https://ieeexplore.ieee.org/document/8237350/)]

* Sparse Representation-Based Open Set Recognition (TPAMI 2017) 
[[Paper](https://ieeexplore.ieee.org/document/7577876/)]

* Towards unsupervised open-set person re-identification (ICIP 2016) 
[[Paper](https://ieeexplore.ieee.org/document/7532461/)]

* Detecting and classifying scars, marks, and tattoos found in the wild (BTAS 2012) 
[[Paper](https://ieeexplore.ieee.org/document/6374555/)]

* Toward Open-Set Face Recognition (CVPRw 2017) 
[[Paper](https://ieeexplore.ieee.org/document/8014819/)]

* Towards Open-World Person Re-Identification by One-Shot Group-Based Verification (TPAMI 2016) 
[[Paper](https://ieeexplore.ieee.org/document/7152932/)]

* Open-World Visual Recognition Using Knowledge Graphs (Arxiv 2017) 
[[Paper](https://arxiv.org/abs/1708.08310)] 

* Identifying Unknown Unknowns in the Open World: Representations and Policies for Guided Exploration (AAAI 2017) 
[[Paper](https://arxiv.org/abs/1610.09064)] 

* Online Open World Recognition (Arxiv 2016) 
[[Paper](https://arxiv.org/abs/1604.02275)] 

<!----------------------------------------------------------------------------------------------------------------------------------------------->
## Novel Class Discovery
<!----------------------------------------------------------------------------------------------------------------------------------------------->
### Surveys
* Novel Class Discovery: an Introduction and Key Concepts (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2302.12028)]

* Deep Class-Incremental Learning: A Survey (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2302.03648)]


### 2023 Papers
#### CVPR
* Open-Category Human-Object Interaction Pre-training via Language Modeling Framework (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Zheng_Open-Category_Human-Object_Interaction_Pre-Training_via_Language_Modeling_Framework_CVPR_2023_paper.html)]

* Modeling Inter-Class and Intra-Class Constraints in Novel Class Discovery (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2210.03591)]<br>
*Datasets: CIFAR-10, CIFAR-100, and ImageNet, CIFAR100-50, CIFAR100-20*<br>
*Task: Image Classification*

* Bootstrap Your Own Prior: Towards Distribution-Agnostic Novel Class Discovery (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Bootstrap_Your_Own_Prior_Towards_Distribution-Agnostic_Novel_Class_Discovery_CVPR_2023_paper.pdf)][[Code](https://github.com/muliyangm)]

* Novel Class Discovery for 3D Point Cloud Semantic Segmentation (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.11610)]

* PromptCAL: Contrastive Affinity Learning via Auxiliary Prompts for Generalized Novel Category Discovery (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2212.05590)][[Code](https://github.com/sheng-eatamath/PromptCAL)]<br>
*Datasets: CIFAR-10, CIFAR-100, and ImageNet-100, CUB-200, SCars, Herbarium19*<br>
*Task: Image Classification*

* Upcycling Models under Domain and Category Shift (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.07110)][[Code](https://github.com/ispc-lab/GLC)]<br>
*Datasets: Office-31, Office-Home, VisDA-C, Domain-Net*<br>
*Task: Image Classification*

* On-the-fly Category Discovery (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Du_On-the-Fly_Category_Discovery_CVPR_2023_paper.pdf)][[Code](https://github.com/PRIS-CV/On-the-fly-Category-Discovery)]

* Dynamic Conceptional Contrastive Learning for Generalized Category Discovery (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.17393)][[Code](https://github.com/TPCD/DCCL)]<br>
*Datasets: CIFAR-10, CIFAR-100, and ImageNet-100, CUB-200, SCars, Oxford-IIIT Pet*<br>
*Task: Image Classification*


<!--#### ICLR
#### NeurIPS-->
#### ICCV
* Class-relation Knowledge Distillation for Novel Class Discovery (ICCV 2023) 
[[Paper](https://arxiv.org/abs/2307.09158)][[Code](https://github.com/kleinzcy/Cr-KD-NCD)]<br>

* Incremental Generalized Category Discovery (ICCV 2023) 
[[Paper](https://arxiv.org/abs/2304.14310)]

* Learning Semi-supervised Gaussian Mixture Models for Generalized Category Discovery (ICCV 2023) 
[[Paper](https://arxiv.org/abs/2305.06144)]

* Parametric Classification for Generalized Category Discovery: A Baseline Study (ICCV 2023) 
[[Paper](https://arxiv.org/abs/2211.11727v2)][[Code](https://github.com/CVMI-Lab/SimGCD)]<br>
*Datasets: CIFAR-10, CIFAR-100, and ImageNet-100, CUB-200, SCars, Herbarium19*<br>
*Task: Image Classification*

#### ICML
* When and How Does Known Class Help Discover Unknown Ones? Provable Understandings Through Spectral Analysis (ICML 2023) 
[[Paper](https://openreview.net/pdf?id=JHodnaW5WZ)]
[[Code](https://github.com/deeplearning-wisc/NSCL)]
<!--#### IEEE-Access
#### ECCV-->
#### AAAI
* Generalized Category Discovery with Decoupled Prototypical Network (AAAI 2023) 
[[Paper](https://arxiv.org/abs/2211.15115)]
[[Code](https://github.com/Lackel/DPN)]<br>
*Datasets: BANKING, StackOverflow, CLINC*<br>
*Task: Intent Classification*
<!--#### TPAMI
#### CVPRw-->
#### WACV
* Scaling Novel Object Detection With Weakly Supervised Detection Transformers (WACV 2023) 
[[Paper](http://arxiv.org/abs/2207.05205)]<br>
*Datasets: Few-Shot  Object  Detection (FSOD), FGVC-Aircraft, iNaturalist  2017, PASCAL VOC 2007*<br>
*Task: Object Detection*
<!--#### IJCV
#### BMVC
#### ICCw
#### WACVw-->


#### Other Journals and Conferences

* Open-world Semi-supervised Novel Class Discovery (IJCAI 2023) 
[[Paper](https://arxiv.org/abs/2305.13095)][[Code](https://github.com/LiuJMzzZ/OpenNCD)]

* NeurNCD: Novel Class Discovery via Implicit Neural Representation (IJCAI 2023) 
[[Paper](https://github.com/jmwang0117/NeurNCD/blob/main/IJCAI_2023.pdf)][[Code](https://github.com/jmwang0117/NeurNCD)]

* Supervised Knowledge May Hurt Novel Class Discovery Performance (TMLR 2023) 
[[Paper](https://openreview.net/forum?id=oqOBTo5uWD&referrer=%5BTMLR%5D(%2Fgroup%3Fid%3DTMLR))][[Code](https://github.com/J-L-O/SK-Hurt-NCD)]

* An Interactive Interface for Novel Class Discovery in Tabular Data (ECML PKDD 2023, Demo Track 2023) 
[[Paper](https://arxiv.org/pdf/2306.12919.pdf)][[Code](https://github.com/ColinTr/InteractiveClustering)]

#### Arxiv & Others


* Effective Cross-instance Positive Relations for Generalized Category Discovery (ICLR 2023 Submission) 
[[Paper](https://openreview.net/forum?id=hag85Gdq_RA)]<br>
*Datasets: CIFAR-10, CIFAR-100, and ImageNet-100, CUB-200, SCars, Herbarium19*<br>
*Task: Image Classification*

* Novel Class Discovery under Unreliable Sampling (ICLR 2023 Submission) 
[[Paper](https://openreview.net/forum?id=uJzSlJruEjk)]<br>
*Datasets: CIFAR-10/100, ImageNet*<br>
*Task: Image Classification*

* Zero-Knowledge Zero-Shot Learning for Novel Visual Category Discovery (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2302.04427v1)]<br>
*Datasets:  Attribute Pascal and Yahoo (APY), Animals with Attributes2 (AWA2), Caltech-UCSD-Birds 200-2011 (CUB), SUN*<br>
*Task: Image Classification*

* PromptCAL: Contrastive Affinity Learning via Auxiliary Prompts for Generalized Novel Category Discovery (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2212.05590)]
[[Code](https://github.com/sheng-eatamath/PromptCAL)]<br>
*Datasets: CIFAR-10/100, CUB-200, StandfordCars, Aircraft, ImageNet-100*<br>
*Task: Image Classification*

* Boosting Novel Category Discovery Over Domains with Soft Contrastive Learning and All-in-One Classifier (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2211.11262)]<br>


* Mutual Information-based Generalized Category Discovery (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2212.00334)]
[[Code](https://github.com/fchiaroni/Mutual-Information-Based-GCD)]<br>
*Datasets: CIFAR-10, CIFAR-100, and ImageNet-100, CUB-200, SCars, Herbarium19*<br>
*Task: Image Classification*


* Large-scale Pre-trained Models are Surprisingly Strong in Incremental Novel Class Discovery (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.15975)][[Code](https://github.com/OatmealLiu/MSc-iNCD)]

* USNID: A Framework for Unsupervised and Semi-supervised New Intent Discovery (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2304.07699)]<br>
*Datasets: BANKING, StackOverflow, CLINC*<br>
*Task: Intent Classification*

* CiPR: An Efficient Framework with Cross-instance Positive Relations for Generalized Category Discovery (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2304.06928)]

* NEV-NCD: Negative Learning, Entropy, and Variance regularization based novel action categories discovery (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2304.07354)][[Code](https://huggingface.co/datasets/mahmed10/MPSC_MV)]

* What's in a Name? Beyond Class Indices for Image Recognition (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2304.02364)]

* CLIP-GCD: Simple Language Guided Generalized Category Discovery (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.10420)]

* Federated Generalized Category Discovery (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.14107v1)]
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

* Novel Class Discovery without Forgetting (ECCV 2022) 
[[Paper](https://arxiv.org/abs/2207.10659)]

* Class-incremental Novel Class Discovery (ECCV 2022) 
[[Paper](https://arxiv.org/abs/2207.08605)]

* Towards Realistic Semi-Supervised Learning (ECCV 2022 Oral) 
[[Paper](https://arxiv.org/abs/2207.02269)]

#### AAAI
* Self-Labeling Framework for Novel Category Discovery over Domains (AAAI 2022) 
[[Paper](https://ojs.aaai.org/index.php/AAAI/article/download/20224/version/18521/19983)]<br>
*Datasets: Office (Amazon, DSLR, Webcam); OfficeHome (art, clipart, product, and real); VisDA (synthetic and real)*<br>
*Task: Image Classification*

<!-- #### TPAMI -->
#### EMNLP

* Fine-grained Category Discovery under Coarse-grained supervision with Hierarchical Weighted Self-contrastive Learning (EMNLP 2022) 
[[Paper](https://arxiv.org/abs/2210.07733v1)]
[[Code](https://github.com/Lackel/Hierarchical_Weighted_SCL)]<br>
*Datasets: CLINC, Web of Science (WOS), HWU64*<br>
*Task: Text Classification*

#### CVPRw
* Spacing Loss for Discovering Novel Categories (CVPRw 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022W/CLVision/papers/Joseph_Spacing_Loss_for_Discovering_Novel_Categories_CVPRW_2022_paper.pdf)]
[[Code](https://github.com/JosephKJ/Awesome-Novel-Class-Discovery)]<br>
*Datasets: CIFAR-10 and CIFAR-100*<br>
*Task: Image Classification*

<!--#### WACV




#### IJCV-->
#### BMVC
* XCon: Learning with Experts for Fine-grained Category Discovery (BMVC 2022) 
[[Paper](https://arxiv.org/abs/2208.01898v1)]
[[Code](https://github.com/YiXXin/XCon)]<br>
*Datasets: CIFAR-10/100, ImageNet-100, CUB-200, Standford Cars, FGVC-Aircraft, and Oxford-IIIT Pet*<br>
*Task: Image Classification*

<!--#### ICCw-->

#### NeurIPSw

* A Closer Look at Novel Class Discovery from the Labeled Set (NeurIPSw 2022) 
[[Paper](https://arxiv.org/abs/2209.09120v4)]<br>
*Datasets: CIFAR100-50, ImageNet*<br>
*Task: Image Classification*

#### WACVw 
* Uncertainty Aware Proposal Segmentation for Unknown Object Detection (WACVw 2022) 
[[Paper](https://openaccess.thecvf.com/content/WACV2022W/DNOW/papers/Li_Uncertainty_Aware_Proposal_Segmentation_for_Unknown_Object_Detection_WACVW_2022_paper.pdf)]

#### Other Journals and Conferences

* A Method for Discovering Novel Classes in Tabular Data (ICKG 2022) 
[[Paper](https://arxiv.org/abs/2209.01217v3)]<br>
*Datasets: ForestCover Type, Letter Recognition, Human Activity Recognition, Satimage, Pen-Based Handwritten Digits Recognition, 1990 US Census Data, MNIST*<br>
*Task: Tabular Data Classification*

* Residual Tuning: Toward Novel Category Discovery Without Labels (TNNLS 2022) 
[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9690577&tag=1)]
[[Code](https://github.com/liuyudut/ResTune)]<br>
*Datasets: CIFAR-10, CIFAR-100, TinyImageNet*<br>
*Task: Image Classification*

* Open Set Domain Adaptation By Novel Class Discovery (ICME 2022) 
[[Paper](https://arxiv.org/abs/2203.03329v1)]<br>
*Datasets: Office-31, Office-Home, DomainNet*<br>
*Task: Image Classification and Domain Adaptation*

* Novel Class Discovery: A Dependency Approach (ICASSP 2022) 
[[Paper](https://ieeexplore.ieee.org/document/9747827/)]

* Discovering Novel Categories in Sar Images in Open Set Conditions (IGARSS 2022) 
[[Paper](https://ieeexplore.ieee.org/document/9883175/)]

#### Arxiv & Others
* Mutual Information-guided Knowledge Transfer for Novel Class Discovery (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2206.12063v1)]<br>
*Datasets: CIFAR10, CIFAR100, OmniGlot, ImageNet*<br>
*Task: Image Classification*


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

* Modeling Inter-Class and Intra-Class Constraints in Novel Class Discovery (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2210.03591v2)]<br>
*Datasets: CIFAR10, CIFAR100, ImageNet*<br>
*Task: Image Classification*

* A Simple Parametric Classification Baseline for Generalized Category Discovery (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2211.11727v1)]
[[Code](https://github.com/CVMI-Lab/SimGCD)]<br>
*Datasets: CIFAR10/100, ImageNet-100, Semantic Shift Benchmark (SSB, including CUB, Stanford Cars, and FGVC-Aircraft), Herbarium 19*<br>
*Task: Image Classification*

<!----------------------------------------------------------------------------------------------------------------------------------------------->
### 2021 Papers
#### CVPR
* OpenMix: Reviving Known Knowledge for Discovering Novel Visual Categories in an Open World (CVPR 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhong_OpenMix_Reviving_Known_Knowledge_for_Discovering_Novel_Visual_Categories_in_CVPR_2021_paper.pdf)]<br>
*Datasets: CIFAR10, CIFAR 100, Imagenet*<br>
*Task: Image Classification*

* Neighborhood Contrastive Learning for Novel Class Discovery (CVPR 2021) 
[[Paper](http://arxiv.org/abs/2106.10731)]<br>
*Datasets: CIFAR10, CIFAR 100, Imagenet*<br>
*Task: Image Classification*

<!--#### ICLR-->
#### NeurIPS
* Novel Visual Category Discovery with Dual Ranking Statistics and Mutual Knowledge Distillation (NeurIPS 2021) 
[[Paper](https://arxiv.org/abs/2107.03358)]
[[Code](https://github.com/DTennant/dual-rank-ncd)]<br>
*Datasets: CIFAR-10, CIFAR-100, ImageNet-1K, ImageNet-100, CUB-200, Stanford-Cars, FGVC aircraft*<br>
*Task: Image Classification*

#### ICCV
* Towards Novel Target Discovery Through Open-Set Domain Adaptation (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Jing_Towards_Novel_Target_Discovery_Through_Open-Set_Domain_Adaptation_ICCV_2021_paper.pdf)]
[[Code](https://github.com/scottjingtt/SROSDA.git)]<br>
*Datasets: D2AwA; IAwA2*<br>
*Task: Image Classification*

* A Unified Objective for Novel Class Discovery (ICCV 2021 Oral) 
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
#### IJCV-->
#### BMVC

* Learning to Generate Novel Classes for Deep Metric Learning (BMVC 2021) 
[[Paper](https://www.bmvc2021-virtualconference.com/assets/papers/0631.pdf)]<br>
*Datasets: CUB-200-2011 (CUB), Cars-196 (Cars), Stanford Online Product (SOP), and In-shop Clothes Retrieval (In-Shop)*<br>
*Task: Image Classification*

<!--#### ICCw-->
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

* Learning to Reconstruct Shapes from Unseen Classes (NeurIPS 2018) 
[[Paper](https://papers.nips.cc/paper/2018/hash/208e43f0e45c4c78cafadb83d2888cb6-Abstract.html)]
[[Code](http://genre.csail.mit.edu/)]<br>
*Datasets: ShapeNet, Pix3D, non-rigid shapes such as humans and horses*<br>
*Task: 3D Shape Reconstruction*

* Memory Replay GANs: Learning to Generate New Categories without Forgetting (NeurIPS 2018) 
[[Paper](https://papers.nips.cc/paper/2018/hash/a57e8915461b83adefb011530b711704-Abstract.html)]
[[Code](https://github.com/WuChenshen/MeRGAN)]<br>
*Datasets: MNIST, SVHN and LSUN*<br>
*Task: Image Generation*

* The continuous categorical: a novel simplex-valued exponential family (ICML 2020) 
[[Paper](https://arxiv.org/abs/2002.08563)]
[[Code](https://github.com/cunningham-lab/cb_and_cc)]<br>
*Datasets: Simulation, 2019 UK general election, MNIST*<br>
*Task: Continuous Categorical Distribution*

* ShapeMask: Learning to Segment Novel Objects by Refining Shape Priors (ICCV 2019) 
[[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kuo_ShapeMask_Learning_to_Segment_Novel_Objects_by_Refining_Shape_Priors_ICCV_2019_paper.pdf)]
[[Code](https://sites.google.com/view/shapemask/home)]<br>
*Datasets: PASCAL VOC, COCO*<br>
*Task: Instance Segmentation*

* A probabilistic constrained clustering for transfer learning and image category discovery (CVPRw 2018) 
[[Paper](https://arxiv.org/abs/1806.11078)]

* Adversarial Joint-Distribution Learning for Novel Class Sketch-Based Image Retrieval (ICCVw 2019) 
[[Paper](https://openaccess.thecvf.com/content_ICCVW_2019/papers/MDALC/Pandey_Adversarial_Joint-Distribution_Learning_for_Novel_Class_Sketch-Based_Image_Retrieval_ICCVW_2019_paper.pdf)]

* Unseen Class Discovery in Open-world Classification (Arxiv 2018) 
[[Paper](https://arxiv.org/abs/1801.05609)]


<!----------------------------------------------------------------------------------------------------------------------------------------------->
## Open Vocabulary
<!----------------------------------------------------------------------------------------------------------------------------------------------->

### 2023 Papers
#### CVPR
* OpenScene: 3D Scene Understanding with Open Vocabularies (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2211.15654)]

* Open-Vocabulary Point-Cloud Object Detection without 3D Annotation (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2304.00788)]

* Learning to Generate Language-supervised and Open-vocabulary Scene Graph using Pre-trained Visual-Semantic Space (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Learning_To_Generate_Language-Supervised_and_Open-Vocabulary_Scene_Graph_Using_Pre-Trained_CVPR_2023_paper.html)]

* Side Adapter Network for Open-Vocabulary Semantic Segmentation (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2302.12242)]

* Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.04803)]

* Mask-free OVIS: Open-Vocabulary Instance Segmentation without Manual Mask Annotations (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.16891)]

* Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2210.04150)]

* Region-Aware Pretraining for Open-Vocabulary Object Detection with Vision Transformers (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Kim_Region-Aware_Pretraining_for_Open-Vocabulary_Object_Detection_With_Vision_Transformers_CVPR_2023_paper.html)]

* Object-Aware Distillation Pyramid for Open-Vocabulary Object Detection (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.05892)]

* Aligning Bag of Regions for Open-Vocabulary Object Detection (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2302.13996)]

* Open-set Fine-grained Retrieval via Prompting Vision-Language Evaluator (CVPR 2023) 
[[Paper](https://readpaper.com/paper/1608794558669368064)]

* Open Vocabulary Semantic Segmentation with Patch Aligned Contrastive Learning (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2212.04994)]

* FreeSeg: Unified, Universal and Open-Vocabulary Image Segmentation (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.17225)]

* GLIGEN: Open-Set Grounded Text-to-Image Generation (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2301.07093)]

* DetCLIPv2: Scalable Open-Vocabulary Object Detection Pre-training via Word-Region Alignment (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Yao_DetCLIPv2_Scalable_Open-Vocabulary_Object_Detection_Pre-Training_via_Word-Region_Alignment_CVPR_2023_paper.html)]

* OvarNet: Towards Open-vocabulary Object Attribute Recognition (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2301.09506)]

* PLA: Language-Driven Open-Vocabulary 3D Scene Understanding (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2211.16312)]

* Open-vocabulary Attribute Detection (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2211.12914)]

* Learning Open-vocabulary Semantic Segmentation Models From Natural Language Supervision (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2301.09121)]

* Learning to Generate Text-grounded Mask for Open-world Semantic Segmentation from Only Image-Text Pairs (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2212.00785)]

* CORA: Adapting CLIP for Open-Vocabulary Detection with Region Prompting and Anchor Pre-Matching (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.13076)]

* OVTrack: Open-Vocabulary Multiple Object Tracking (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Li_OVTrack_Open-Vocabulary_Multiple_Object_Tracking_CVPR_2023_paper.html)]

* Learning to Detect and Segment for Open Vocabulary Object Detection (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2212.12130)]

* Learning to Detect and Segment for Open Vocabulary Object Detection (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2212.12130)]


#### ICLR
* Open-vocabulary Object Detection via Vision and Language Knowledge Distillation	(ICLR 2023) 
[[Paper](https://openreview.net/forum?id=lL3lnMbR4WU)]
[[Code](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild)]<br>
*Datasets: LVIS, PASCAL VOC, COCO, Objects365*<br>
*Task: Object Detection*
<!--#### NeurIPS-->

#### ICCV

* Global Knowledge Calibration for Fast Open-Vocabulary Segmentation (ICCV 2023) 
[[Paper](https://arxiv.org/abs/2303.09181)]

* Open-vocabulary Panoptic Segmentation with Embedding Modulation (ICCV 2023) 
[[Paper](https://arxiv.org/abs/2303.11324)]

#### ICML
* SegCLIP: Patch Aggregation with Learnable Centers for Open-Vocabulary Semantic Segmentation (ICML 2023) 
[[Paper](https://arxiv.org/abs/2211.14813)]

* Open-VCLIP: Transforming CLIP to an Open-vocabulary Video Model via Interpolated Weight Optimization (ICML 2023) 
[[Paper](https://arxiv.org/abs/2302.00624)]

* Open-Vocabulary Universal Image Segmentation with MaskCLIP (ICML 2023) 
[[Paper](http://arxiv.org/abs/2208.08984)]

* Multi-Modal Classifiers for Open-Vocabulary Object Detection (ICML 2023) 
[[Paper](http://arxiv.org/abs/2207.01887)]

<!--#### IEEE-Access
#### ECCV
#### AAAI
#### TPAMI-->
#### CVPRw
* Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models (CVPRw 2023) 
[[Paper](https://arxiv.org/abs/2303.04803)]
<!--#### WACV
#### IJCV
#### BMVC
#### ICCw -->
#### Arxiv & Others

* A Language-Guided Benchmark for Weakly Supervised Open Vocabulary Semantic Segmentation	(Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2302.14163)]

* Aligning Bag of Regions for Open-Vocabulary Object Detection (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2302.13996)]

* From Occlusion to Insight: Object Search in Semantic Shelves using Large Language Models (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2302.12915)]

* Side Adapter Network for Open-Vocabulary Semantic Segmentation (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2302.12242)]

* CHiLS: Zero-Shot Image Classification with Hierarchical Label Sets (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2302.02551)]

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

<!--#### ICLR-->
#### NeurIPS
* Patching open-vocabulary models by interpolating weights (NeurIPS 2022) 
[[Paper](https://openreview.net/forum?id=uOQNvEfjpaC)]
[[Code](https://github.com/mlfoundations/patching)]<br>
*Datasets: Cars, DTD, EuroSAT, GTSRB, KITTI, MNIST, RESISC45, SUN397, and SVHN. We use the remaining tasks as supported tasks: CIFAR10, CIFAR100, Food101, ImageNet, and STL10*<br>
*Task: Model Patching*

* Bridging the Gap between Object and Image-level Representations for Open-Vocabulary Detection (NeurIPS 2022) 
[[Paper](https://openreview.net/forum?id=aKXBrj0DHm)]
[[Code](https://github.com/hanoonaR/object-centric-ovd)]<br>
*Datasets: COCO, LVIS v1.0, OpenImages, Objects365*<br>
*Task: Object Detection*

* Paraphrasing Is All You Need for Novel Object Captioning (NeurIPS 2022) 
[[Paper](https://arxiv.org/abs/2209.12343)]<br>
*Datasets:  Open Images V4, COCO Captions 2017*<br>
*Task: Image Captioning*

<!--#### ICCV
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

* Open Vocabulary Electroencephalography-to-Text Decoding and Zero-Shot Sentiment Classification (AAAI 2022) 
[[Paper](https://arxiv.org/abs/2112.02690)]
[[Code](https://github.com/MikeWangWZHL/EEG-To-Text)]<br>
*Datasets: ZuCo*<br>
*Task: Brain Signals Language Decoding*

<!--#### TPAMI
#### CVPRw-->
#### WACV

* From Node To Graph: Joint Reasoning on Visual-Semantic Relational Graph for Zero-Shot Detection (WACV 2022) 
[[Paper](https://openaccess.thecvf.com/content/WACV2022/papers/Nie_From_Node_To_Graph_Joint_Reasoning_on_Visual-Semantic_Relational_Graph_WACV_2022_paper.pdf)]
[[Code](https://github.com/witnessai)]<br>
*Datasets: MSCOCO*<br>
*Task: Object Detection*

* Trading-Off Information Modalities in Zero-Shot Classification (WACV 2022) 
[[Paper](https://openaccess.thecvf.com/content/WACV2022/papers/Sanchez_Trading-Off_Information_Modalities_in_Zero-Shot_Classification_WACV_2022_paper.pdf)]
[[Code](http://github.com/jadrs/zsl)]<br>
*Datasets: Caltech UCSD Birds 200-2011 (CUB), Animals with Attributes 1 and 2 (AWA1 & AWA2), attribute Pascal & Yahoo (APY), SUN attributes (SUN) and Oxford flowers (FLO)*<br>
*Task: Image Classification*

<!--#### IJCV-->
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
### Older Papers
* A Latent Morphology Model for Open-Vocabulary Neural Machine Translation (ICLR 2020 Spotlight) 
[[Paper](https://openreview.net/forum?id=BJxSI1SKDH)]
[[Code](https://github.com/d-ataman/lmm)]<br>
*Datasets: Arabic (AR), Czech (CS) and Turkish (TR)*<br>
*Task: Neural Machine Translation*

* Open Vocabulary Learning on Source Code with a Graph-Structured Cache (ICML 2019) 
[[Paper](https://arxiv.org/abs/1810.08305)]<br>
*Datasets: Java source code*<br>
*Task: Java source code Learning*

* Visual Question Generation for Class Acquisition of Unknown Objects (ECCV 2018) 
[[Paper](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Kohei_Uehara_Visual_Question_Generation_ECCV_2018_paper.pdf)]
[[Code](https://github.com/mil-tokyo/vqg-unknown)]<br>
*Datasets: Visual Genome, ILSVRC2012, ILSVRC2010, WordNet*<br>
*Task: Visual Question Generation, Object Detection*

* Jointly Discovering Visual Objects and Spoken Words from Raw Sensory Input (ECCV 2018) 
[[Paper](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/David_Harwath_Jointly_Discovering_Visual_ECCV_2018_paper.pdf)]
[[Code](http://groups.csail.mit.edu/sls/downloads/placesaudio/)]<br>
*Datasets: Places Audio Caption, ADE20k, MSCOCO*<br>
*Task: Audio-Visual Associative Localizations*

* Image Captioning with Unseen Objects (BMVC 2019) 
[[Paper](https://bmvc2019.org/wp-content/papers/0124.html)]<br>
*Datasets: COCO*<br>
*Task: Image Captioning*

* nocaps: novel object captioning at scale (ICCV 2019) 
[[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Agrawal_nocaps_novel_object_captioning_at_scale_ICCV_2019_paper.pdf)]
[[Code](https://nocaps.org)]<br>
*Datasets: nocaps, COCO Captions*<br>
*Task: Image Captioning*

* Pointing Novel Objects in Image Captioning (CVPR 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Pointing_Novel_Objects_in_Image_Captioning_CVPR_2019_paper.pdf)]<br>
*Datasets: held-out COCO, ImageNet*<br>
*Task: Image Captioning*

* Learning User Representations for Open Vocabulary Image Hashtag Prediction (CVPR 2020) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Durand_Learning_User_Representations_for_Open_Vocabulary_Image_Hashtag_Prediction_CVPR_2020_paper.pdf)]<br>
*Datasets: YFCC100M*<br>
*Task: Image Hashtag Prediction*

* Open-Edit: Open-Domain Image Manipulation with Open-Vocabulary Instructions (ECCV 2020) 
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560086.pdf)]
[[Code](https://github.com/xh-liu/Open-Edit)]<br>
*Datasets: BSDS500, Conceptual Captions*<br>
*Task: Image Manipulation*

<!----------------------------------------------------------------------------------------------------------------------------------------------->
## Fine Grained
<!----------------------------------------------------------------------------------------------------------------------------------------------->

### 2023 Papers
#### CVPR

* Dynamic Coarse-to-Fine Learning for Oriented Tiny Object Detection (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.01219)]

* DeCo: Decomposition and Reconstruction for Compositional Temporal Grounding via Coarse-to-Fine Contrastive Ranking (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_DeCo_Decomposition_and_Reconstruction_for_Compositional_Temporal_Grounding_via_Coarse-To-Fine_CVPR_2023_paper.html)]

* Exploiting Unlabelled Photos for Stronger Fine-Grained SBIR (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.13779)]

* Fine-Grained Face Swapping via Regional GAN Inversion (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2211.14068)]

* Learning Attribute and Class-Specific Representation Duet for Fine-Grained Fashion Analysis (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Jiao_Learning_Attribute_and_Class-Specific_Representation_Duet_for_Fine-Grained_Fashion_Analysis_CVPR_2023_paper.html)]

* Learning Common Rationale to Improve Self-Supervised Representation for Fine-Grained Visual Recognition Problems (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.01669)]

* FashionSAP: Symbols and Attributes Prompt for Fine-grained Fashion Vision-Language Pre-training (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Han_FashionSAP_Symbols_and_Attributes_Prompt_for_Fine-Grained_Fashion_Vision-Language_Pre-Training_CVPR_2023_paper.html)]

* Fine-grained Image-text Matching by Cross-modal Hard Aligning Network (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2008.05231)]

* Open-set Fine-grained Retrieval via Prompting Vision-Language Evaluator (CVPR 2023) 
[[Paper](https://readpaper.com/paper/1608794558669368064)]

* Weakly Supervised Posture Mining for Fine-grained Classification (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Tang_Weakly_Supervised_Posture_Mining_for_Fine-Grained_Classification_CVPR_2023_paper.html)]

* Hierarchical Fine-Grained Image Forgery Detection and Localization (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.17111)]

* Coreset Sampling from Open-Set for Fine-Grained Self-Supervised Learning (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.11101)]

* CLIP for All Things Zero-Shot Sketch-Based Image Retrieval, Fine-Grained or Not (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.13440)]

* Fine-Grained Classification with Noisy Labels (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.02404)]

* An Erudite Fine-Grained Visual Classification Model (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Chang_An_Erudite_Fine-Grained_Visual_Classification_Model_CVPR_2023_paper.html)]

* Fine-Grained Classification with Noisy Labels (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.02404)]


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
#### CVPRw-->
#### WACV

* CAST: Conditional Attribute Subsampling Toolkit for Fine-Grained Evaluation (WACV 2023) 
[[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Robbins_CAST_Conditional_Attribute_Subsampling_Toolkit_for_Fine-Grained_Evaluation_WACV_2023_paper.pdf)]<br>
*Datasets: WebFace42M, CAST-11*<br>
*Task: Face Recognition*

* SSFE-Net: Self-Supervised Feature Enhancement for Ultra-Fine-Grained Few-Shot Class Incremental Learning (WACV 2023) 
[[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Pan_SSFE-Net_Self-Supervised_Feature_Enhancement_for_Ultra-Fine-Grained_Few-Shot_Class_Incremental_Learning_WACV_2023_paper.pdf)]<br>
*Datasets: CottonCultivar, SoyCultivarLocal, PlantVillage, Caltech-UCSD Birds-200-2011 (CUB200), Mini-ImageNet*<br>
*Task: Image Classification*


* Mixture Outlier Exposure: Towards Out-of-Distribution Detection in Fine-Grained Environments (WACV 2023) 
[[Paper](http://arxiv.org/abs/2106.03917)]
[[Code](https://github.com/zjysteven/MixOE)]<br>
*Datasets: WebVision 1.0*<br>
*Task: Out-of-Distribution Detection Images*

<!--#### IJCV
#### BMVC
#### ICCw-->
#### Arxiv & Others
* Combining Photogrammetric Computer Vision and Semantic Segmentation for Fine-Grained Understanding of Coral Reef Growth Under Climate Change (WACVw 2023) 
[[Paper](http://arxiv.org/abs/2212.04132)]

* Test-Time Amendment with a Coarse Classifier for Fine-Grained Classification (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2302.00368)]
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

#### ICLR

* A Fine-Grained Analysis on Distribution Shift (ICLR 2022 Oral) 
[[Paper](https://openreview.net/forum?id=Dl4LetuLdyK)]<br>

* Superclass-Conditional Gaussian Mixture Model For Learning Fine-Grained Embeddings (ICLR 2022 Spotlight) 
[[Paper](https://openreview.net/forum?id=vds4SNooOe)]<br>

* Contrastive Fine-grained Class Clustering via Generative Adversarial Networks (ICLR 2022 Spotlight) 
[[Paper](https://openreview.net/forum?id=XWODe7ZLn8f)]<br>

* FILIP: Fine-grained Interactive Language-Image Pre-Training (ICLR 2022) 
[[Paper](https://openreview.net/forum?id=cpDhcsEDC2)]<br>

#### NeurIPS

* Relational Proxies: Emergent Relationships as Fine-Grained Discriminators (NeurIPS 2022) 
[[Paper](https://openreview.net/forum?id=xONqm0NUJc)]<br>

* Fine-Grained Semantically Aligned Vision-Language Pre-Training (NeurIPS 2022) 
[[Paper](https://openreview.net/forum?id=yam42JWePu)]<br>

* Structuring Uncertainty for Fine-Grained Sampling in Stochastic Segmentation Networks (NeurIPS 2022) 
[[Paper](https://openreview.net/forum?id=odOQU9PYrkD)]<br>

* Learning Efficient Vision Transformers via Fine-Grained Manifold Distillation (NeurIPS 2022) 
[[Paper](https://openreview.net/forum?id=crFMP5irwzn)]<br>

<!-- #### ICCV
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


#### AAAI

* TransFG: A Transformer Architecture for Fine-Grained Recognition (AAAI 2022) 
[[Paper](https://aaai-2022.virtualchair.net/poster_aaai8475)]

* Category-Specific Nuance Exploration Network for Fine-Grained Object Retrieval (AAAI 2022) 
[[Paper](https://aaai-2022.virtualchair.net/poster_AAAI4346.html)]

* Dual Attention Networks for Few-Shot Fine-Grained Recognition (AAAI 2022) 
[[Paper](https://aaai-2022.virtualchair.net/poster_AAAI1537.html)]

* Hyperbolic Disentangled Representation for Fine-Grained Aspect Extraction (AAAI 2022) 
[[Paper](https://aaai-2022.virtualchair.net/poster_AAAI6574.html)]

* Class Guided Channel Weighting Network for Fine-Grained Semantic Segmentation (AAAI 2022) 
[[Paper](https://aaai-2022.virtualchair.net/poster_AAAI6621.html)]

* Exploiting Fine-Grained Face Forgery Clues via Progressive Enhancement Learning (AAAI 2022) 
[[Paper](https://aaai-2022.virtualchair.net/poster_AAAI1011.html)]

* Towards Fine-Grained Reasoning for Fake News Detection (AAAI 2022) 
[[Paper](https://aaai-2022.virtualchair.net/poster_AAAI12332.html)]

* Accurate and Scalable Gaussian Processes for Fine-Grained Air Quality Inference (AAAI 2022) 
[[Paper](https://aaai-2022.virtualchair.net/poster_AISI11178.html)]

* Fine-Grained Urban Flow Inference via Normalizing Flow (Student Abstract) (AAAI 2022) 
[[Paper](https://aaai-2022.virtualchair.net/poster_SA418.html)]

* Pinpointing Fine-Grained Relationships between Hateful Tweets and Replies (AAAI 2022) 
[[Paper](https://aaai-2022.virtualchair.net/poster_AAAI6353.html)]

* COVID-EENet: Predicting Fine-Grained Impact of COVID-19 on Local Economies (AAAI 2022) 
[[Paper](https://aaai-2022.virtualchair.net/poster_AISI1959.html)]

<!--#### TPAMI
#### CVPRw-->
#### WACV
* FalCon: Fine-Grained Feature Map Sparsity Computing With Decomposed Convolutions for Inference Optimization (WACV 2022) 
[[Paper](https://openaccess.thecvf.com/content/WACV2022/papers/Xu_FalCon_Fine-Grained_Feature_Map_Sparsity_Computing_With_Decomposed_Convolutions_for_WACV_2022_paper.pdf)]<br>
*Datasets: CIFAR-10, CIFAR-100, ILSVRC-2012*<br>
*Task: Inference Optimization*

* 3DRefTransformer: Fine-Grained Object Identification in Real-World Scenes Using Natural Language (WACV 2022) 
[[Paper](https://openaccess.thecvf.com/content/WACV2022/papers/Abdelreheem_3DRefTransformer_Fine-Grained_Object_Identification_in_Real-World_Scenes_Using_Natural_Language_WACV_2022_paper.pdf)]
[[Code](https://vision-cair.github.io/3dreftransformer/)]<br>
*Datasets: Nr3D, Sr3D*<br>
*Task: 3D Object Identification*

<!--#### IJCV-->
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
#### CVPR
* Learning Deep Classifiers Consistent With Fine-Grained Novelty Detection (CVPR 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Cheng_Learning_Deep_Classifiers_Consistent_With_Fine-Grained_Novelty_Detection_CVPR_2021_paper.pdf)]<br>
*Datasets: small- and large-scale FGVC*<br>
*Task: Novelty Detection*

* Graph-Based High-Order Relation Discovery for Fine-Grained Recognition (CVPR 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhao_Graph-Based_High-Order_Relation_Discovery_for_Fine-Grained_Recognition_CVPR_2021_paper.pdf)]

* Learning Deep Classifiers Consistent With Fine-Grained Novelty Detection (CVPR 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Cheng_Learning_Deep_Classifiers_Consistent_With_Fine-Grained_Novelty_Detection_CVPR_2021_paper.pdf)]

* More Photos Are All You Need: Semi-Supervised Learning for Fine-Grained Sketch Based Image Retrieval (CVPR 2021) 
[[Paper](http://arxiv.org/abs/2103.13990)]

* RefineMask: Towards High-Quality Instance Segmentation With Fine-Grained Features (CVPR 2021) 
[[Paper](http://arxiv.org/abs/2104.08569)]

* A Realistic Evaluation of Semi-Supervised Learning for Fine-Grained Classification (CVPR 2021) 
[[Paper](http://arxiv.org/abs/2104.00679)]

* Your "Flamingo" is My "Bird": Fine-Grained, or Not (CVPR 2021) 
[[Paper](http://arxiv.org/abs/2011.09040)]

* FaceSec: A Fine-Grained Robustness Evaluation Framework for Face Recognition Systems (CVPR 2021) 
[[Paper](http://arxiv.org/abs/2104.04107)]

* Discrimination-Aware Mechanism for Fine-Grained Representation Learning (CVPR 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_Discrimination-Aware_Mechanism_for_Fine-Grained_Representation_Learning_CVPR_2021_paper.pdf)]

* Fine-Grained Angular Contrastive Learning With Coarse Labels (CVPR 2021) 
[[Paper](http://arxiv.org/abs/2012.03515)]

* Neural Prototype Trees for Interpretable Fine-Grained Image Recognition (CVPR 2021) 
[[Paper](http://arxiv.org/abs/2012.02046)]

* Learning Fine-Grained Segmentation of 3D Shapes Without Part Labels (CVPR 2021) 
[[Paper](http://arxiv.org/abs/2103.13030)]

#### ICLR

* Learning N:M Fine-grained Structured Sparse Neural Networks From Scratch (ICLR 2021) 
[[Paper](https://openreview.net/forum?id=K9bw7vqp_s)]<br>

#### NeurIPS
* A2-Net: Learning Attribute-Aware Hash Codes for Large-Scale Fine-Grained Image Retrieval (NeurIPS 2021 Spotlight) 
[[Paper](https://openreview.net/forum?id=mIki_kyHpLb)]<br>

* Fine-Grained Zero-Shot Learning with DNA as Side Information (NeurIPS 2021) 
[[Paper](https://openreview.net/forum?id=RqAzAoL8BER)]<br>

* Weak-shot Fine-grained Classification via Similarity Transfer (NeurIPS 2021) 
[[Paper](https://openreview.net/forum?id=wDJUUcCTNI2)]<br>

* Landmark-RxR: Solving Vision-and-Language Navigation with Fine-Grained Alignment Supervision (NeurIPS 2021) 
[[Paper](https://openreview.net/forum?id=41QJ--DLjoD)]<br>

#### ICCV

* Persistent Homology Based Graph Convolution Network for Fine-Grained 3D Shape Segmentation (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wong_Persistent_Homology_Based_Graph_Convolution_Network_for_Fine-Grained_3D_Shape_ICCV_2021_paper.pdf)]

* Webly Supervised Fine-Grained Recognition: Benchmark Datasets and an Approach (ICCV 2021) 
[[Paper](http://arxiv.org/abs/2108.02399)]

* Semi-Supervised Single-Stage Controllable GANs for Conditional Fine-Grained Image Generation (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Semi-Supervised_Single-Stage_Controllable_GANs_for_Conditional_Fine-Grained_Image_Generation_ICCV_2021_paper.pdf)]

* Benchmark Platform for Ultra-Fine-Grained Visual Categorization Beyond Human Performance (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Yu_Benchmark_Platform_for_Ultra-Fine-Grained_Visual_Categorization_Beyond_Human_Performance_ICCV_2021_paper.pdf)]

* Variational Feature Disentangling for Fine-Grained Few-Shot Classification (ICCV 2021) 
[[Paper](http://arxiv.org/abs/2010.03255)]

* CrackFormer: Transformer Network for Fine-Grained Crack Detection (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_CrackFormer_Transformer_Network_for_Fine-Grained_Crack_Detection_ICCV_2021_paper.pdf)]

* Talk-To-Edit: Fine-Grained Facial Editing via Dialog (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Jiang_Talk-To-Edit_Fine-Grained_Facial_Editing_via_Dialog_ICCV_2021_paper.pdf)]

* N-ImageNet: Towards Robust, Fine-Grained Object Recognition With Event Cameras (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_N-ImageNet_Towards_Robust_Fine-Grained_Object_Recognition_With_Event_Cameras_ICCV_2021_paper.pdf)]

* Fine-Grained Semantics-Aware Representation Enhancement for Self-Supervised Monocular Depth Estimation (ICCV 2021) 
[[Paper](http://arxiv.org/abs/2108.08829)]

* ALADIN: All Layer Adaptive Instance Normalization for Fine-Grained Style Similarity (ICCV 2021) 
[[Paper](http://arxiv.org/abs/2103.09776)]

* Grafit: Learning Fine-Grained Image Representations With Coarse Labels (ICCV 2021) 
[[Paper](http://arxiv.org/abs/2011.12982)]

* Stochastic Partial Swap: Enhanced Model Generalization and Interpretability for Fine-Grained Recognition (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Huang_Stochastic_Partial_Swap_Enhanced_Model_Generalization_and_Interpretability_for_Fine-Grained_ICCV_2021_paper.pdf)]

* Learning Canonical 3D Object Representation for Fine-Grained Recognition (ICCV 2021) 
[[Paper](http://arxiv.org/abs/2108.04628)]


<!--#### ICML
#### IEEE-Access
#### ECCV
#### AAAI
#### TPAMI-->
#### CVPRw
* Localized Triplet Loss for Fine-Grained Fashion Image Retrieval (CVPRw 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021W/CVFAD/papers/DInnocente_Localized_Triplet_Loss_for_Fine-Grained_Fashion_Image_Retrieval_CVPRW_2021_paper.pdf)]

* Fine-Grained Visual Attribute Extraction From Fashion Wear (CVPRw 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021W/CVFAD/papers/Parekh_Fine-Grained_Visual_Attribute_Extraction_From_Fashion_Wear_CVPRW_2021_paper.pdf)]

* CLIP-Art: Contrastive Pre-Training for Fine-Grained Art Classification (CVPRw 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021W/CVFAD/papers/Conde_CLIP-Art_Contrastive_Pre-Training_for_Fine-Grained_Art_Classification_CVPRW_2021_paper.pdf)]

#### WACV

* Multi-Modal Reasoning Graph for Scene-Text Based Fine-Grained Image Classification and Retrieval (WACV 2021) 
[[Paper](http://arxiv.org/abs/2009.09809)]

* Text-to-Image Generation Grounded by Fine-Grained User Attention (WACV 2021) 
[[Paper](http://arxiv.org/abs/2011.03775)]

* TB-Net: A Three-Stream Boundary-Aware Network for Fine-Grained Pavement Disease Segmentation (WACV 2021) 
[[Paper](https://openaccess.thecvf.com/content/WACV2021/papers/Zhang_TB-Net_A_Three-Stream_Boundary-Aware_Network_for_Fine-Grained_Pavement_Disease_Segmentation_WACV_2021_paper.pdf)]

* Coarse- and Fine-Grained Attention Network With Background-Aware Loss for Crowd Density Map Estimation (WACV 2021) 
[[Paper](https://openaccess.thecvf.com/content/WACV2021/papers/Rong_Coarse-_and_Fine-Grained_Attention_Network_With_Background-Aware_Loss_for_Crowd_WACV_2021_paper.pdf)]

* Fair Comparison: Quantifying Variance in Results for Fine-Grained Visual Categorization (WACV 2021) 
[[Paper](https://openaccess.thecvf.com/content/WACV2021/papers/Gwilliam_Fair_Comparison_Quantifying_Variance_in_Results_for_Fine-Grained_Visual_Categorization_WACV_2021_paper.pdf)]

* Fine-Grained Foreground Retrieval via Teacher-Student Learning (WACV 2021) 
[[Paper](https://openaccess.thecvf.com/content/WACV2021/papers/Wu_Fine-Grained_Foreground_Retrieval_via_Teacher-Student_Learning_WACV_2021_paper.pdf)]

* Intra-Class Part Swapping for Fine-Grained Image Classification (WACV 2021) 
[[Paper](https://openaccess.thecvf.com/content/WACV2021/papers/Zhang_Intra-Class_Part_Swapping_for_Fine-Grained_Image_Classification_WACV_2021_paper.pdf)]

* Towards Enhancing Fine-Grained Details for Image Matting (WACV 2021) 
[[Paper](https://openaccess.thecvf.com/content/WACV2021/papers/Liu_Towards_Enhancing_Fine-Grained_Details_for_Image_Matting_WACV_2021_paper.pdf)]

<!--#### IJCV-->
#### BMVC

* The Curious Layperson: Fine-Grained Image Recognition without Expert Labels (BMVC 2021) 
[[Paper](https://www.bmvc2021-virtualconference.com/assets/papers/0229.pdf)]

* Human Attention in Fine-grained Classification (BMVC 2021) 
[[Paper](https://www.bmvc2021-virtualconference.com/assets/papers/0421.pdf)]

* Feature Fusion Vision Transformer for Fine-Grained Visual Categorization (BMVC 2021) 
[[Paper](https://www.bmvc2021-virtualconference.com/assets/papers/0685.pdf)]

* Fine-grained Multi-Modal Self-Supervised Learning (BMVC 2021) 
[[Paper](https://www.bmvc2021-virtualconference.com/assets/papers/0702.pdf)]

* Channel DropBlock: An Improved Regularization Method for Fine-Grained Visual Classification (BMVC 2021) 
[[Paper](https://www.bmvc2021-virtualconference.com/assets/papers/0601.pdf)]

* SwinFGHash: Fine-grained Image Retrieval via Transformer-based Hashing Network (BMVC 2021) 
[[Paper](https://www.bmvc2021-virtualconference.com/assets/papers/1551.pdf)]

#### ICCw

* Fine-Grain Prediction of Strawberry Freshness Using Subsurface Scattering (ICCVw 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021W/LFFAI/papers/Klotz_Fine-Grain_Prediction_of_Strawberry_Freshness_Using_Subsurface_Scattering_ICCVW_2021_paper.pdf)]

<!--#### Arxiv & Others-->

<!----------------------------------------------------------------------------------------------------------------------------------------------->
### 2020 Papers
#### CVPR

* Fine-Grained Generalized Zero-Shot Learning via Dense Attribute-Based Attention (CVPR 2020) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Huynh_Fine-Grained_Generalized_Zero-Shot_Learning_via_Dense_Attribute-Based_Attention_CVPR_2020_paper.pdf)]

* Progressive Adversarial Networks for Fine-Grained Domain Adaptation (CVPR 2020) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Progressive_Adversarial_Networks_for_Fine-Grained_Domain_Adaptation_CVPR_2020_paper.pdf)]

* Interpretable and Accurate Fine-grained Recognition via Region Grouping (CVPR 2020) 
[[Paper](http://arxiv.org/abs/2005.10411)]

* Fine-Grained Image-to-Image Transformation Towards Visual Recognition (CVPR 2020) 
[[Paper](http://arxiv.org/abs/2001.03856)]

* Weakly Supervised Fine-Grained Image Classification via Guassian Mixture Model Oriented Discriminative Learning (CVPR 2020) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Weakly_Supervised_Fine-Grained_Image_Classification_via_Guassian_Mixture_Model_Oriented_CVPR_2020_paper.pdf)]

* Say As You Wish: Fine-Grained Control of Image Caption Generation With Abstract Scene Graphs (CVPR 2020) 
[[Paper](http://arxiv.org/abs/2003.00387)]

* Attention Convolutional Binary Neural Tree for Fine-Grained Visual Categorization (CVPR 2020) 
[[Paper](http://arxiv.org/abs/1909.11378)]

* Sketch Less for More: On-the-Fly Fine-Grained Sketch-Based Image Retrieval (CVPR 2020) 
[[Paper](http://arxiv.org/abs/2002.10310)]

* 3D Part Guided Image Editing for Fine-Grained Object Understanding (CVPR 2020) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_3D_Part_Guided_Image_Editing_for_Fine-Grained_Object_Understanding_CVPR_2020_paper.pdf)]

* Solving Mixed-Modal Jigsaw Puzzle for Fine-Grained Sketch-Based Image Retrieval (CVPR 2020) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Pang_Solving_Mixed-Modal_Jigsaw_Puzzle_for_Fine-Grained_Sketch-Based_Image_Retrieval_CVPR_2020_paper.pdf)]


#### NeurIPS

* Fine-Grained Dynamic Head for Object Detection (NeurIPS 2020) 
[[Paper](https://papers.nips.cc/paper_files/paper/2020/hash/7f6caf1f0ba788cd7953d817724c2b6e-Abstract.html)]<br>

* No Subclass Left Behind: Fine-Grained Robustness in Coarse-Grained Classification Problems (NeurIPS 2020) 
[[Paper](https://papers.nips.cc/paper_files/paper/2020/hash/e0688d13958a19e087e123148555e4b4-Abstract.html)]<br>

* Compositional Zero-Shot Learning via Fine-Grained Dense Feature Composition (NeurIPS 2020) 
[[Paper](https://papers.nips.cc/paper_files/paper/2020/hash/e58cc5ca94270acaceed13bc82dfedf7-Abstract.html)]<br>

#### ECCV

* ReferIt3D: Neural Listeners for Fine-Grained 3D Object Identification in Real-World Scenes (ECCV 2020) 
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460409.pdf)]

* ExchNet: A Unified Hashing Network for Large-Scale Fine-Grained Image Retrieval (ECCV 2020) 
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480188.pdf)]

* Self-supervising Fine-grained Region Similarities for Large-scale Image Localization (ECCV 2020) 
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490358.pdf)]

* Beyond 3DMM Space: Towards Fine-grained 3D Face Reconstruction (ECCV 2020) 
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530341.pdf)]

* Classes Matter: A Fine-grained Adversarial Approach to Cross-domain Semantic Segmentation (ECCV 2020) 
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590630.pdf)]

* SceneSketcher: Fine-Grained Image Retrieval with Scene Sketches (ECCV 2020) 
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640698.pdf)]

* Fine-Grained Visual Classification via Progressive Multi-Granularity Training of Jigsaw Patches (ECCV 2020) 
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650154.pdf)]

* OneGAN: Simultaneous Unsupervised Learning of Conditional Image Generation, Foreground Segmentation, and Fine-Grained Clustering (ECCV 2020) 
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710511.pdf)]

* Toward Fine-grained Facial Expression Manipulation (ECCV 2020) 
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730035.pdf)]


#### CVPRw

* ePillID Dataset: A Low-Shot Fine-Grained Benchmark for Pill Identification (CVPRw 2020) 
[[Paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w54/Usuyama_ePillID_Dataset_A_Low-Shot_Fine-Grained_Benchmark_for_Pill_Identification_CVPRW_2020_paper.pdf)]

* Towards Fine-Grained Sampling for Active Learning in Object Detection (CVPRw 2020) 
[[Paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w54/Desai_Towards_Fine-Grained_Sampling_for_Active_Learning_in_Object_Detection_CVPRW_2020_paper.pdf)]

* Focus Longer to See Better: Recursively Refined Attention for Fine-Grained Image Classification (CVPRw 2020) 
[[Paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w51/Shroff_Focus_Longer_to_See_Better_Recursively_Refined_Attention_for_Fine-Grained_CVPRW_2020_paper.pdf)]

* Fine-Grained Recognition in High-Throughput Phenotyping (CVPRw 2020) 
[[Paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w5/Lyu_Fine-Grained_Recognition_in_High-Throughput_Phenotyping_CVPRW_2020_paper.pdf)]

#### WACV


* L\*ReLU: Piece-wise Linear Activation Functions for Deep Fine-grained Visual Categorization (WACV 2020) 
[[Paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Basirat_LReLU_Piece-wise_Linear_Activation_Functions_for_Deep_Fine-grained_Visual_Categorization_WACV_2020_paper.pdf)]

* An Adversarial Domain Adaptation Network for Cross-Domain Fine-Grained Recognition (WACV 2020) 
[[Paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Wang_An_Adversarial_Domain_Adaptation_Network_for_Cross-Domain_Fine-Grained_Recognition_WACV_2020_paper.pdf)]

* ELoPE: Fine-Grained Visual Classification with Efficient Localization, Pooling and Embedding (WACV 2020) 
[[Paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Hanselmann_ELoPE_Fine-Grained_Visual_Classification_with_Efficient_Localization_Pooling_and_Embedding_WACV_2020_paper.pdf)]

* Color Composition Similarity and Its Application in Fine-grained Similarity (WACV 2020) 
[[Paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Ha_Color_Composition_Similarity_and_Its_Application_in_Fine-grained_Similarity_WACV_2020_paper.pdf)]

* Fine-grained Image Classification and Retrieval by Combining Visual and Locally Pooled Textual Features (WACV 2020) 
[[Paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Mafla_Fine-grained_Image_Classification_and_Retrieval_by_Combining_Visual_and_Locally_WACV_2020_paper.pdf)]

#### BMVC

* Cross-Modal Hierarchical Modelling for Fine-Grained Sketch Based Image Retrieval (BMVC 2020) 
[[Paper](https://www.bmvc2020-conference.com/assets/papers/0102.pdf)]

* Contrastively-reinforced Attention Convolutional Neural Network for Fine-grained Image Recognition (BMVC 2020) 
[[Paper](https://www.bmvc2020-conference.com/assets/papers/0656.pdf)]

* On the Exploration of Incremental Learning for Fine-grained Image Retrieval (BMVC 2020) 
[[Paper](https://www.bmvc2020-conference.com/assets/papers/0079.pdf)]
<!----------------------------------------------------------------------------------------------------------------------------------------------->
### 2019 Papers

#### CVPR
* Generalising Fine-Grained Sketch-Based Image Retrieval (CVPR 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pang_Generalising_Fine-Grained_Sketch-Based_Image_Retrieval_CVPR_2019_paper.pdf)]

* PartNet: A Large-Scale Benchmark for Fine-Grained and Hierarchical Part-Level 3D Object Understanding (CVPR 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Mo_PartNet_A_Large-Scale_Benchmark_for_Fine-Grained_and_Hierarchical_Part-Level_3D_CVPR_2019_paper.pdf)]

* FSA-Net: Learning Fine-Grained Structure Aggregation for Head Pose Estimation From a Single Image (CVPR 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_FSA-Net_Learning_Fine-Grained_Structure_Aggregation_for_Head_Pose_Estimation_From_CVPR_2019_paper.pdf)]

* Weakly Supervised Complementary Parts Models for Fine-Grained Image Classification From the Bottom Up (CVPR 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ge_Weakly_Supervised_Complementary_Parts_Models_for_Fine-Grained_Image_Classification_From_CVPR_2019_paper.pdf)]

* Distilling Object Detectors With Fine-Grained Feature Imitation (CVPR 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Distilling_Object_Detectors_With_Fine-Grained_Feature_Imitation_CVPR_2019_paper.pdf)]

* Looking for the Devil in the Details: Learning Trilinear Attention Sampling Network for Fine-Grained Image Recognition (CVPR 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zheng_Looking_for_the_Devil_in_the_Details_Learning_Trilinear_Attention_CVPR_2019_paper.pdf)]

* Destruction and Construction Learning for Fine-Grained Image Recognition (CVPR 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Destruction_and_Construction_Learning_for_Fine-Grained_Image_Recognition_CVPR_2019_paper.pdf)]

* FineGAN: Unsupervised Hierarchical Disentanglement for Fine-Grained Object Generation and Discovery (CVPR 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Singh_FineGAN_Unsupervised_Hierarchical_Disentanglement_for_Fine-Grained_Object_Generation_and_Discovery_CVPR_2019_paper.pdf)]

* Multispectral Imaging for Fine-Grained Recognition of Powders on Complex Backgrounds (CVPR 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhi_Multispectral_Imaging_for_Fine-Grained_Recognition_of_Powders_on_Complex_Backgrounds_CVPR_2019_paper.pdf)]

* Interpretable and Fine-Grained Visual Explanations for Convolutional Neural Networks (CVPR 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wagner_Interpretable_and_Fine-Grained_Visual_Explanations_for_Convolutional_Neural_Networks_CVPR_2019_paper.pdf)]

* PartNet: A Recursive Part Decomposition Network for Fine-Grained and Hierarchical Shape Segmentation (CVPR 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yu_PartNet_A_Recursive_Part_Decomposition_Network_for_Fine-Grained_and_Hierarchical_CVPR_2019_paper.pdf)]

* 3D Guided Fine-Grained Face Manipulation (CVPR 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Geng_3D_Guided_Fine-Grained_Face_Manipulation_CVPR_2019_paper.pdf)]


#### ICLR

* Coarse-grain Fine-grain Coattention Network for Multi-evidence Question Answering (ICLR 2019) 
[[Paper](https://openreview.net/forum?id=Syl7OsRqY7)]<br>
*Datasets: TriviaQA, WikiHop*<br>
*Task: Question Answering*

#### NeurIPS


* Meta-Reinforced Synthetic Data for One-Shot Fine-Grained Visual Recognition (NeurIPS 2019) 
[[Paper](https://papers.nips.cc/paper_files/paper/2019/hash/418ef6127e44214882c61e372e866691-Abstract.html)]<br>

* Learning Deep Bilinear Transformation for Fine-grained Image Representation (NeurIPS 2019) 
[[Paper](https://papers.nips.cc/paper_files/paper/2019/hash/959ef477884b6ac2241b19ee4fb776ae-Abstract.html)]<br>

* Fine-grained Optimization of Deep Neural Networks (NeurIPS 2019) 
[[Paper](https://papers.nips.cc/paper_files/paper/2019/hash/c2626d850c80ea07e7511bbae4c76f4b-Abstract.html)]<br>

#### ICCV

* Fine-Grained Segmentation Networks: Self-Supervised Segmentation for Improved Long-Term Visual Localization (ICCV 2019) 
[[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Larsson_Fine-Grained_Segmentation_Networks_Self-Supervised_Segmentation_for_Improved_Long-Term_Visual_Localization_ICCV_2019_paper.pdf)]

* Cost-Aware Fine-Grained Recognition for IoTs Based on Sequential Fixations (ICCV 2019) 
[[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Cost-Aware_Fine-Grained_Recognition_for_IoTs_Based_on_Sequential_Fixations_ICCV_2019_paper.pdf)]

* Adversarial Fine-Grained Composition Learning for Unseen Attribute-Object Recognition (ICCV 2019) 
[[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wei_Adversarial_Fine-Grained_Composition_Learning_for_Unseen_Attribute-Object_Recognition_ICCV_2019_paper.pdf)]

* Person-in-WiFi: Fine-Grained Person Perception Using WiFi (ICCV 2019) 
[[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Person-in-WiFi_Fine-Grained_Person_Perception_Using_WiFi_ICCV_2019_paper.pdf)]

* Selective Sparse Sampling for Fine-Grained Image Recognition (ICCV 2019) 
[[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_Selective_Sparse_Sampling_for_Fine-Grained_Image_Recognition_ICCV_2019_paper.pdf)]

* SkyScapes Fine-Grained Semantic Understanding of Aerial Scenes (ICCV 2019) 
[[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Azimi_SkyScapes__Fine-Grained_Semantic_Understanding_of_Aerial_Scenes_ICCV_2019_paper.pdf)]

* Cross-X Learning for Fine-Grained Visual Categorization (ICCV 2019) 
[[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Luo_Cross-X_Learning_for_Fine-Grained_Visual_Categorization_ICCV_2019_paper.pdf)]

* Learning a Mixture of Granularity-Specific Experts for Fine-Grained Categorization (ICCV 2019) 
[[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Learning_a_Mixture_of_Granularity-Specific_Experts_for_Fine-Grained_Categorization_ICCV_2019_paper.pdf)]

* Presence-Only Geographical Priors for Fine-Grained Image Classification (ICCV 2019) 
[[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Aodha_Presence-Only_Geographical_Priors_for_Fine-Grained_Image_Classification_ICCV_2019_paper.pdf)]

* Detecting 11K Classes: Large Scale Object Detection Without Fine-Grained Bounding Boxes (ICCV 2019) 
[[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yang_Detecting_11K_Classes_Large_Scale_Object_Detection_Without_Fine-Grained_Bounding_ICCV_2019_paper.pdf)]

#### CVPRw

* Interpreting Fine-Grained Dermatological Classification by Deep Learning (CVPRw 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/ISIC/Mishra_Interpreting_Fine-Grained_Dermatological_Classification_by_Deep_Learning_CVPRW_2019_paper.pdf)]

#### BMVC

* Guided Zoom: Questioning Network Evidence for Fine-grained Classification (BMVC 2019) 
[[Paper](https://bmvc2019.org/wp-content/papers/0061.html)]<br>
*Datasets: CaltechUCSD (CUB-200-2011) Birds, Stanford Dogs, FGVC-Aircraft*<br>
*Task: Image Classification*

* Group Based Deep Shared Feature Learning for Fine-grained Image Classification (BMVC 2019) 
[[Paper](https://bmvc2019.org/wp-content/papers/0885.html)]
[[Code](https://github.com/xueluli/GSFL-Net)]<br>
*Datasets: CaltechUCSD (CUB-200-2011) Birds, Stanford Dogs, FGVC-Aircraft*<br>
*Task: Image Classification*

#### ICCVw

* A Refined 3D Pose Dataset for Fine-Grained Object Categories (ICCVw 2019) 
[[Paper](https://openaccess.thecvf.com/content_ICCVW_2019/papers/R6D/Wang_A_Refined_3D_Pose_Dataset_for_Fine-Grained_Object_Categories_ICCVW_2019_paper.pdf)]

* Count, Crop and Recognise: Fine-Grained Recognition in the Wild (ICCVw 2019) 
[[Paper](https://openaccess.thecvf.com/content_ICCVW_2019/papers/CVWC/Bain_Count_Crop_and_Recognise_Fine-Grained_Recognition_in_the_Wild_ICCVW_2019_paper.pdf)]

* Geo-Aware Networks for Fine-Grained Recognition (ICCVw 2019) 
[[Paper](https://openaccess.thecvf.com/content_ICCVW_2019/papers/CVWC/Chu_Geo-Aware_Networks_for_Fine-Grained_Recognition_ICCVW_2019_paper.pdf)]

* ELPephants: A Fine-Grained Dataset for Elephant Re-Identification (ICCVw 2019) 
[[Paper](https://openaccess.thecvf.com/content_ICCVW_2019/papers/CVWC/Korschens_ELPephants_A_Fine-Grained_Dataset_for_Elephant_Re-Identification_ICCVW_2019_paper.pdf)]

* Squeezed Bilinear Pooling for Fine-Grained Visual Categorization (ICCVw 2019) 
[[Paper](https://openaccess.thecvf.com/content_ICCVW_2019/papers/SDL-CV/Liao_Squeezed_Bilinear_Pooling_for_Fine-Grained_Visual_Categorization_ICCVW_2019_paper.pdf)]

<!----------------------------------------------------------------------------------------------------------------------------------------------->
### Older Papers
* AttnGAN: Fine-Grained Text to Image Generation With Attentional Generative Adversarial Networks (CVPR 2018) 
[[Paper](http://arxiv.org/abs/1711.10485)]

* Large Scale Fine-Grained Categorization and Domain-Specific Transfer Learning (CVPR 2018) 
[[Paper](http://arxiv.org/abs/1806.06193)]

* Learning a Discriminative Filter Bank Within a CNN for Fine-Grained Recognition (CVPR 2018) 
[[Paper](http://arxiv.org/abs/1611.09932)]

* Webly Supervised Learning Meets Zero-Shot Learning: A Hybrid Approach for Fine-Grained Classification (CVPR 2018) 
[[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Niu_Webly_Supervised_Learning_CVPR_2018_paper.pdf)]

* Stacked Semantics-Guided Attention Model for Fine-Grained Zero-Shot Learning (NeurIPS 2018) 
[[Paper](https://papers.nips.cc/paper/2018/hash/9087b0efc7c7acd1ef7e153678809c77-Abstract.html)]<br>
*Datasets: CUB and NABird*<br>
*Task: Image Classification*

* Attend and Rectify: a gated attention mechanism for fine-grained recovery (ECCV 2018) 
[[Paper](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Pau_Rodriguez_Lopez_Attend_and_Rectify_ECCV_2018_paper.pdf)]

* DeepKSPD: Learning Kernel-matrix-based SPD Representation for Fine-grained Image Recognition (ECCV 2018) 
[[Paper](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Melih_Engin_DeepKSPD_Learning_Kernel-matrix-based_ECCV_2018_paper.pdf)]

* Fine-Grained Visual Categorization using Meta-Learning Optimization with Sample Selection of Auxiliary Data (ECCV 2018) 
[[Paper](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Yabin_Zhang_Fine-Grained_Visual_Categorization_ECCV_2018_paper.pdf)]

* Hierarchical Bilinear Pooling for Fine-Grained Visual Recognition (ECCV 2018) 
[[Paper](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Chaojian_Yu_Hierarchical_Bilinear_Pooling_ECCV_2018_paper.pdf)]

* Grassmann Pooling as Compact Homogeneous Bilinear Pooling for Fine-Grained Visual Classification (ECCV 2018) 
[[Paper](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Xing_Wei_Grassmann_Pooling_for_ECCV_2018_paper.pdf)]

* Learning to Navigate for Fine-grained Classification (ECCV 2018) 
[[Paper](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Ze_Yang_Learning_to_Navigate_ECCV_2018_paper.pdf)]

* Pairwise Confusion for Fine-Grained Visual Classification (ECCV 2018) 
[[Paper](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Abhimanyu_Dubey_Improving_Fine-Grained_Visual_ECCV_2018_paper.pdf)]

* Multi-Attention Multi-Class Constraint for Fine-grained Image Recognition (ECCV 2018) 
[[Paper](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Ming_Sun_Multi-Attention_Multi-Class_Constraint_ECCV_2018_paper.pdf)]

* Cross-Domain Hallucination Network for Fine-Grained Object Recognition (CVPRw 2018) 
[[Paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w21/Lin_Cross-Domain_Hallucination_Network_CVPR_2018_paper.pdf)]


<!----------------------------------------------------------------------------------------------------------------------------------------------->
## Long Tail
<!----------------------------------------------------------------------------------------------------------------------------------------------->
### Survey 
* Deep Long-Tailed Learning: A Survey (Arxiv 2021) 
[[Paper](https://arxiv.org/abs/2110.04596)]

### 2023 Papers
#### CVPR
* Class-Conditional Sharpness-Aware Minimization for Deep Long-Tailed Recognition (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Zhou_Class-Conditional_Sharpness-Aware_Minimization_for_Deep_Long-Tailed_Recognition_CVPR_2023_paper.html)]

* Global and Local Mixture Consistency Cumulative Learning for Long-tailed Visual Recognitions (CVPR 2023) 
[[Paper](https://paperswithcode.com/paper/global-and-local-mixture-consistency)]

* Balancing Logit Variation for Long-tailed Semantic Segmentation (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Balancing_Logit_Variation_for_Long-Tailed_Semantic_Segmentation_CVPR_2023_paper.html)]

* Transfer Knowledge from Head to Tail: Uncertainty Calibration under Long-tailed Distribution (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Transfer_Knowledge_From_Head_to_Tail_Uncertainty_Calibration_Under_Long-Tailed_CVPR_2023_paper.html)]

* Towards Realistic Long-Tailed Semi-Supervised Learning: Consistency Is All You Need (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Wei_Towards_Realistic_Long-Tailed_Semi-Supervised_Learning_Consistency_Is_All_You_Need_CVPR_2023_paper.html)]

* Rethinking Image Super Resolution from Long-Tailed Distribution Learning Perspective (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Gou_Rethinking_Image_Super_Resolution_From_Long-Tailed_Distribution_Learning_Perspective_CVPR_2023_paper.html)]

* Long-Tailed Visual Recognition via Self-Heterogeneous Integration with Knowledge Excavation (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2304.01279)]

* FCC: Feature Clusters Compression for Long-Tailed Visual Recognition (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Li_FCC_Feature_Clusters_Compression_for_Long-Tailed_Visual_Recognition_CVPR_2023_paper.html)]

* Curvature-Balanced Feature Manifold Learning for Long-Tailed Classification (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.12307)]

* Balanced Product of Calibrated Experts for Long-Tailed Recognition (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2206.05260)]

* SuperDisco: Super-Class Discovery Improves Visual Recognition for the Long-Tail (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2304.00101)]

* No One Left Behind: Improving the Worst Categories in Long-Tailed Learning (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.03630)]


#### ICLR
* CUDA: Curriculum of Data Augmentation for Long-tailed Recognition (ICLR 2023 top 25%) 
[[Paper](https://openreview.net/forum?id=RgUPdudkWlN)]<br>
*Datasets: CIFAR-100-LT, ImageNet-LT, iNaturalist 2018*<br>
*Task: Image Classification*

* Decoupled Training for Long-Tailed Classification With Stochastic Representations (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=bcYZwYo-0t)]<br>

* Long-Tailed Partial Label Learning via Dynamic Rebalancing (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=sXfWoK4KvSW)]<br>

* Long-Tailed Learning Requires Feature Learning (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=S-h1oFv-mq)]<br>

* LPT: Long-tailed Prompt Tuning for Image Classification (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=8pOVAeo8ie)]<br>

* Delving into Semantic Scale Imbalance (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=07tc5kKRIo)]<br>

* On the Effectiveness of Out-of-Distribution Data in Self-Supervised Long-Tail Learning. (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=v8JIQdiN9Sh)]<br>

* Temperature Schedules for self-supervised contrastive methods on long-tail data (ICLR 2023) 
[[Paper](https://openreview.net/forum?id=ejHUr4nfHhD)]<br>

<!--#### NeurIPS
#### ICCV
#### ICML
#### IEEE-Access
#### ECCV
#### AAAI
#### TPAMI
#### CVPRw-->
#### WACV
* Dynamic Re-Weighting for Long-Tailed Semi-Supervised Learning (WACV 2023) 
[[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Peng_Dynamic_Re-Weighting_for_Long-Tailed_Semi-Supervised_Learning_WACV_2023_paper.pdf)]<br>
*Datasets: CIFAR-10, CIFAR-100, ImageNet127*<br>
*Task: Image Classification*

* Difficulty-Net: Learning To Predict Difficulty for Long-Tailed Recognition (WACV 2023) 
[[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Sinha_Difficulty-Net_Learning_To_Predict_Difficulty_for_Long-Tailed_Recognition_WACV_2023_paper.pdf)]
[[Code](https://github.com/hitachi-rd-cv/Difficulty_Net)]<br>
*Datasets: CIFAR100-LT, ImageNet-LT, Places-LT*<br>
*Task: Image Classification*

* Mutual Learning for Long-Tailed Recognition (WACV 2023) 
[[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Park_Mutual_Learning_for_Long-Tailed_Recognition_WACV_2023_paper.pdf)]<br>
*Datasets:  CIFAR100-LT,ImageNet-LT, and iNaturalist 2018*<br>
*Task: Image Classification*

<!--#### IJCV
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
[[Code](https://github.com/Vision-CAIR/RelTransformer)]<br>
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
*Datasets: CIFAR-10-LT & CIFAR-100-LT; ImageNet-LT; iNaturalist; Places-LT*<br>
*Task: Image Classification*

#### ICLR
* Optimal Transport for Long-Tailed Recognition with Learnable Cost Matrix (ICLR 2022) 
[[Paper](https://openreview.net/forum?id=t98k9ePQQpn)]<br>

#### NeurIPS
* Self-Supervised Aggregation of Diverse Experts for Test-Agnostic Long-Tailed Recognition (NeurIPS 2022) 
[[Paper](https://arxiv.org/abs/2107.09249)]
[[Code](https://github.com/Vanint/SADE-AgnosticLT)]<br>
*Datasets: ImageNet-LT, CIFAR100-LT, Places-LT, iNaturalist 2018*<br>
*Task: Image Classification*

<!--#### ICCV-->
#### ICML
* Open-Sampling: Exploring Out-of-Distribution Data for Re-balancing long-tailed Datasets (ICML 2022) 
[[Paper](https://proceedings.mlr.press/v162/wei22c/wei22c.pdf)]
[[Code](https://github.com/hongxin001/logitnorm_ood)]<br>
*Datasets: long-tailed CIFAR10/100, CelebA-5, Places-LT*<br>
*Task: Out-of-Distribution Detection, Image Classification*

* Partial and Asymmetric Contrastive Learning for Out-of-Distribution Detection in Long-Tailed Recognition (ICML 2022) 
[[Paper](https://arxiv.org/abs/2207.01160)]
[[Code](https://github.com/amazon-research/long-tailed-ood-detection)]<br>
*Datasets: CIFAR10-LT, CIFAR100-LT, and ImageNet-LT*<br>
*Task: Image Classification*

* AdAUC: End-to-end Adversarial AUC Optimization Against Long-tail Problems (ICML 2022) 
[[Paper](https://arxiv.org/abs/2206.12169)]<br>
*Datasets: CIFAR10-LT, CIFAR100-LT, and MNIST-LT*<br>
*Task: AUC optimization*


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

#### AAAI

* LUNA: Localizing Unfamiliarity Near Acquaintance for Open-Set Long-Tailed Recognition (AAAI 2022)
[[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/19887)]<br>
*Datasets: ImageNet-LT, Places-LT, Marine Species (MS)-LT*<br>
*Task: Image Classification*

* Imagine by Reasoning: A Reasoning-Based Implicit Semantic Data Augmentation for Long-Tailed Classification (AAAI 2022)
[[Paper](https://arxiv.org/abs/2112.07928)]

* Adaptive Logit Adjustment Loss for Long-Tailed Visual Recognition (AAAI 2022)
[[Paper](https://arxiv.org/abs/2104.06094)]

* Cross-Domain Empirical Risk Minimization for Unbiased Long-Tailed Classification (AAAI 2022)
[[Paper](https://arxiv.org/abs/2112.14380)]

* Memory-Based Jitter: Improving Visual Recognition on Long-Tailed Data with Diversity In Memory (AAAI 2022)
[[Paper](https://arxiv.org/abs/2008.09809)]

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
#### IJCV-->
#### BMVC
* Class-Balanced Loss Based on ClassVolume for Long-Tailed Object Recognition (BMVC 2022) 
[[Paper](https://bmvc2022.mpi-inf.mpg.de/0896.pdf)]<br>
*Datasets: CIFAR-LT, ImageNet-LT, Places-LT, and iNaturalist 2018*<br>
*Task: Image Classification*

* Unleashing the Potential of Vision-Language Models for Long-Tailed Visual Recognition (BMVC 2022) 
[[Paper](https://bmvc2022.mpi-inf.mpg.de/0481.pdf)]<br>
*Datasets: ImageNet-LT, Places-LT, and iNaturalist 2018, Conceptual 12M (CC12M), Conceptual Captions 3M (CC3M) and SBU Captions*<br>
*Task: Image Classification*
<!--#### ICCw
#### Arxiv & Others-->

<!----------------------------------------------------------------------------------------------------------------------------------------------->
### 2021 Papers
#### CVPR

* Contrastive Learning Based Hybrid Networks for Long-Tailed Image Classification (CVPR 2021) 
[[Paper](http://arxiv.org/abs/2103.14267)]

* PML: Progressive Margin Loss for Long-Tailed Age Classification (CVPR 2021) 
[[Paper](http://arxiv.org/abs/2103.02140)]

* Adaptive Class Suppression Loss for Long-Tail Object Detection (CVPR 2021) 
[[Paper](http://arxiv.org/abs/2104.00885)]

* Adversarial Robustness Under Long-Tailed Distribution (CVPR 2021) 
[[Paper](http://arxiv.org/abs/2104.02703)]

* MetaSAug: Meta Semantic Augmentation for Long-Tailed Visual Recognition (CVPR 2021) 
[[Paper](http://arxiv.org/abs/2103.12579)]

* Disentangling Label Distribution for Long-Tailed Visual Recognition (CVPR 2021) 
[[Paper](http://arxiv.org/abs/2012.00321)]

* Conceptual 12M: Pushing Web-Scale Image-Text Pre-Training To Recognize Long-Tail Visual Concepts (CVPR 2021) 
[[Paper](http://arxiv.org/abs/2102.08981)]

* Distribution Alignment: A Unified Framework for Long-Tail Visual Recognition (CVPR 2021) 
[[Paper](http://arxiv.org/abs/2103.16370)]

* Seesaw Loss for Long-Tailed Instance Segmentation (CVPR 2021) 
[[Paper](http://arxiv.org/abs/2008.10032)]

* Unsupervised Discovery of the Long-Tail in Instance Segmentation Using Hierarchical Self-Supervision (CVPR 2021) 
[[Paper](http://arxiv.org/abs/2104.01257)]

* Equalization Loss v2: A New Gradient Balance Approach for Long-Tailed Object Detection (CVPR 2021) 
[[Paper](http://arxiv.org/abs/2012.08548)]

* Long-Tailed Multi-Label Visual Recognition by Collaborative Training on Uniform and Re-Balanced Samplings (CVPR 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Guo_Long-Tailed_Multi-Label_Visual_Recognition_by_Collaborative_Training_on_Uniform_and_CVPR_2021_paper.pdf)]

* Improving Calibration for Long-Tailed Recognition (CVPR 2021) 
[[Paper](http://arxiv.org/abs/2104.00466)]


#### ICLR

* Long-tail learning via logit adjustment (ICLR 2021 Spotlight) 
[[Paper](https://openreview.net/forum?id=37nvvqkCo5)]<br>

* Long-tailed Recognition by Routing Diverse Distribution-Aware Experts (ICLR 2021 Spotlight) 
[[Paper](https://openreview.net/forum?id=D9I3drBz4UC)]<br>

#### NeurIPS

* On Model Calibration for Long-Tailed Object Detection and Instance Segmentation (NeurIPS 2021) 
[[Paper](https://papers.nips.cc/paper_files/paper/2021/hash/14ad095ecc1c3e1b87f3c522836e9158-Abstract.html)]

* Towards Calibrated Model for Long-Tailed Visual Recognition from Prior Perspective (NeurIPS 2021) 
[[Paper](https://papers.nips.cc/paper_files/paper/2021/hash/39ae2ed11b14a4ccb41d35e9d1ba5d11-Abstract.html)]

* Handling Long-tailed Feature Distribution in AdderNets (NeurIPS 2021) 
[[Paper](https://papers.nips.cc/paper_files/paper/2021/hash/95323660ed2124450caaac2c46b5ed90-Abstract.html)]

#### ICCV
* Exploring Long Tail Visual Relationship Recognition With Large Vocabulary (ICCV 2021) 
[[Paper](http://arxiv.org/abs/2004.00436)]

* Exploring Classification Equilibrium in Long-Tailed Object Detection (ICCV 2021) 
[[Paper](http://arxiv.org/abs/2108.07507)]

* Distilling Virtual Examples for Long-Tailed Recognition (ICCV 2021) 
[[Paper](http://arxiv.org/abs/2103.15042)]

* Distributional Robustness Loss for Long-Tail Learning (ICCV 2021) 
[[Paper](http://arxiv.org/abs/2104.03066)]

* FASA: Feature Augmentation and Sampling Adaptation for Long-Tailed Instance Segmentation (ICCV 2021) 
[[Paper](http://arxiv.org/abs/2102.12867)]

* ACE: Ally Complementary Experts for Solving Long-Tailed Recognition in One-Shot (ICCV 2021) 
[[Paper](http://arxiv.org/abs/2108.02385)]

* MosaicOS: A Simple and Effective Use of Object-Centric Images for Long-Tailed Object Detection (ICCV 2021) 
[[Paper](http://arxiv.org/abs/2102.08884)]

* GistNet: A Geometric Structure Transfer Network for Long-Tailed Recognition (ICCV 2021) 
[[Paper](http://arxiv.org/abs/2105.00131)]

* Self Supervision to Distillation for Long-Tailed Visual Recognition (ICCV 2021) 
[[Paper](http://arxiv.org/abs/2109.04075)]


#### ICML
* Image-Level or Object-Level? A Tale of Two Resampling Strategies for Long-Tailed Detection (ICML 2021) 
[[Paper](http://proceedings.mlr.press/v139/chang21c.html)]

<!--#### IEEE-Access 
#### ECCV
#### AAAI
#### TPAMI-->
#### CVPRw

* Long-Tailed Recognition of SAR Aerial View Objects by Cascading and Paralleling Experts (CVPRw 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Yang_Long-Tailed_Recognition_of_SAR_Aerial_View_Objects_by_Cascading_and_CVPRW_2021_paper.pdf)]

#### WACV

* Towards Resolving the Challenge of Long-Tail Distribution in UAV Images for Object Detection (WACV 2021) 
[[Paper](http://arxiv.org/abs/2011.03822)]

* From Generalized Zero-Shot Learning to Long-Tail With Class Descriptors (WACV 2021) 
[[Paper](http://arxiv.org/abs/2004.02235)]

<!--#### IJCV-->
#### BMVC

* Class-Balanced Distillation for Long-Tailed Visual Recognition (BMVC 2021) 
[[Paper](https://www.bmvc2021-virtualconference.com/assets/papers/0614.pdf)]<br>
*Datasets: ImageNet-LT, iNaturalist17 and iNaturalist18*<br>
*Task: Image Classification*

<!--#### ICCw
#### Arxiv & Others-->
<!----------------------------------------------------------------------------------------------------------------------------------------------->
### 2020 Papers
#### CVPR

* Inflated Episodic Memory With Region Self-Attention for Long-Tailed Visual Recognition (CVPR 2020) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhu_Inflated_Episodic_Memory_With_Region_Self-Attention_for_Long-Tailed_Visual_Recognition_CVPR_2020_paper.pdf)]

* Deep Representation Learning on Long-Tailed Data: A Learnable Embedding Augmentation Perspective (CVPR 2020) 
[[Paper](http://arxiv.org/abs/2002.10826)]

* Domain Balancing: Face Recognition on Long-Tailed Domains (CVPR 2020) 
[[Paper](http://arxiv.org/abs/2003.13791)]

* Rethinking Class-Balanced Methods for Long-Tailed Visual Recognition From a Domain Adaptation Perspective (CVPR 2020) 
[[Paper](http://arxiv.org/abs/2003.10780)]

* Overcoming Classifier Imbalance for Long-Tail Object Detection With Balanced Group Softmax (CVPR 2020) 
[[Paper](http://arxiv.org/abs/2006.10408)]

* BBN: Bilateral-Branch Network With Cumulative Learning for Long-Tailed Visual Recognition (CVPR 2020) 
[[Paper](http://arxiv.org/abs/1912.02413)]

* Equalization Loss for Long-Tailed Object Recognition (CVPR 2020) 
[[Paper](http://arxiv.org/abs/2003.05176)]

#### ICLR

* Decoupling Representation and Classifier for Long-Tailed Recognition (ICLR 2020) 
[[Paper](https://openreview.net/forum?id=r1gRTCVFvB)]
[[Code](github facebookresearch/classifier-balancing)]<br>
*Datasets: AwA, CUB-200-2011, ImageNet, ImageNet-LT, Places, Places-LT, iNaturalist*<br>
*Task: Image Classification*

#### NeurIPS

* Long-Tailed Classification by Keeping the Good and Removing the Bad Momentum Causal Effect (NeurIPS 2020) 
[[Paper](https://papers.nips.cc/paper_files/paper/2020/hash/1091660f3dff84fd648efe31391c5524-Abstract.html)]

* Balanced Meta-Softmax for Long-Tailed Visual Recognition (NeurIPS 2020) 
[[Paper](https://papers.nips.cc/paper_files/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html)]

#### ECCV

* Distribution-Balanced Loss for Multi-Label Classification in Long-Tailed Datasets (ECCV 2020) 
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490154.pdf)]

* Learning From Multiple Experts: Self-paced Knowledge Distillation for Long-tailed Classification (ECCV 2020) 
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500239.pdf)]

* Solving Long-tailed Recognition with Deep Realistic Taxonomic Classifier (ECCV 2020) 
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530171.pdf)]

* The Devil is in Classification: A Simple Framework for Long-tail Instance Segmentation (ECCV 2020) 
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590715.pdf)]

* Feature Space Augmentation for Long-Tailed Data (ECCV 2020) 
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740681.pdf)]

<!----------------------------------------------------------------------------------------------------------------------------------------------->
### Older Papers
* Large-Scale Long-Tailed Recognition in an Open World (CVPR 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Large-Scale_Long-Tailed_Recognition_in_an_Open_World_CVPR_2019_paper.pdf)]

* Unequal-Training for Deep Face Recognition With Long-Tailed Noisy Data (CVPR 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhong_Unequal-Training_for_Deep_Face_Recognition_With_Long-Tailed_Noisy_Data_CVPR_2019_paper.pdf)]

* FastDraw: Addressing the Long Tail of Lane Detection by Adapting a Sequential Prediction Network (CVPRw 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/Vision%20for%20All%20Seasons%20Bad%20Weather%20and%20Nighttime/Philion_FastDraw_Addressing_the_Long_Tail_of_Lane_Detection_by_Adapting_CVPRW_2019_paper.pdf)]

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
#### CVPRw-->
#### WACV

* Anomaly Clustering: Grouping Images Into Coherent Clusters of Anomaly Types (WACV 2023) 
[[Paper](http://arxiv.org/abs/2112.11573)]<br>
*Datasets: MVTec dataset, magnetic tiledefect (MTD) dataset*

* No Shifted Augmentations (NSA): Compact Distributions for Robust Self-Supervised Anomaly Detection (WACV 2023) 
[[Paper](http://arxiv.org/abs/2203.10344)]
[[Code]](https://github.com/IntuitionMachines/NSA)<br>
*Datasets: CIFAR10*

* Training Auxiliary Prototypical Classifiers for Explainable Anomaly Detection in Medical Image Segmentation (WACV 2023) 
[[Paper](http://arxiv.org/abs/2202.11660)]<br>
*Datasets: magnetic resonance (MR), M&Ms challenge dataset, M&Ms-2, PROSTATEx, PROMISE12*

* Anomaly Detection in 3D Point Clouds Using Deep Geometric Descriptors (WACV 2023) 
[[Paper](http://arxiv.org/abs/2203.10344)]<br>
*Datasets: MVTec 3D Anomaly Detection*<br>

* Asymmetric Student-Teacher Networks for Industrial Anomaly Detection (WACV 2023) 
[[Paper](http://arxiv.org/abs/2210.07829)]
[[Code](https://github.com/marco-rudolph/ast2593)]<br>
*Datasets: MVTec 3D Anomaly Detection*

* GLAD: A Global-to-Local Anomaly Detector (WACV 2023) 
[[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Artola_GLAD_A_Global-to-Local_Anomaly_Detector_WACV_2023_paper.pdf)]<br>
*Datasets: MVTec*

* Zero-Shot Versus Many-Shot: Unsupervised Texture Anomaly Detection (WACV 2023) 
[[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Aota_Zero-Shot_Versus_Many-Shot_Unsupervised_Texture_Anomaly_Detection_WACV_2023_paper.pdf)]
[[Code](https://drive.google.com/drive/folders/10OyPzvI3H6llCZBxKxFlKWt1Pw1tkMK1)]<br>
*Datasets: MVTec*

* Image-Consistent Detection of Road Anomalies As Unpredictable Patches (WACV 2023) 
[[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Vojir_Image-Consistent_Detection_of_Road_Anomalies_As_Unpredictable_Patches_WACV_2023_paper.pdf)]<br>
*Datasets: Lost-and-Found(LaF), Road  Anomaly (RA), Road  Obstacles(RO) and Fishyscapes (FS), SegmentMeIfYouCan (SMIYC), CityScapes and BDD100k*

<!--#### IJCV
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
#### ICCV-->
#### ICML
* Latent Outlier Exposure for Anomaly Detection with Contaminated Data (ICML 2022) 
[[Paper](https://arxiv.org/abs/2202.08088)]
[[Code](https://github.com/boschresearch/LatentOE-AD.git)]<br>
*Datasets:  CIFAR-10, Fashion-MNIST, MVTEC, 30 tabular data sets, UCSD Peds1*<br>

* Deep Variational Graph Convolutional Recurrent Network for Multivariate Time Series Anomaly Detection (ICML 2022) 
[[Paper](https://proceedings.mlr.press/v162/chen22x.html)]
[[Code](http://github.com/BoChenGroup/DVGCRN)]<br>
*Datasets:  DND, KPI, SMD, MSL, SMAP*<br>

* FITNESS: (Fine Tune on New and Similar Samples) to detect anomalies in streams with drift and outliers (ICML 2022) 
[[Paper](https://proceedings.mlr.press/v162/sankararaman22a.html)]
[[Code](http://github.com/BoChenGroup/DVGCRN)]<br>
*Datasets: Satellite and Thyroid, IoT Attack, Telemetry*<br>

* Rethinking Graph Neural Networks for Anomaly Detection (ICML 2022) 
[[Paper](https://arxiv.org/abs/2205.15508)]
[[Code](https://github.com/squareRoot3/Rethinking-Anomaly-Detection)]<br>
*Datasets: Amazon, YelpChi, T-Finance, T-Social*<br>
<!--#### IEEE-Access-->
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

* AnoDDPM: Anomaly Detection With Denoising Diffusion Probabilistic Models Using Simplex Noise (CVPRw 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Wyatt_AnoDDPM_Anomaly_Detection_With_Denoising_Diffusion_Probabilistic_Models_Using_Simplex_CVPRW_2022_paper.pdf)]
[[Code](https://github.com/Julian-Wyatt/AnoDDPM)]<br>
*Datasets: MVTec AD*<br>

#### WACV
* One-Class Learned Encoder-Decoder Network With Adversarial Context Masking for Novelty Detection (WACV 2022) 
[[Paper](https://openaccess.thecvf.com/content/WACV2022/papers/Jewell_One-Class_Learned_Encoder-Decoder_Network_With_Adversarial_Context_Masking_for_Novelty_WACV_2022_paper.pdf)]
[[Code](https://github.com/jewelltaylor/OLED)]<br>
*Datasets: MNIST, CIFAR-10, UCSD*<br>
*Task: Novelty Detection, Anomaly*

* CFLOW-AD: Real-Time Unsupervised Anomaly Detection With Localization via Conditional Normalizing Flows (WACV 2022) 
[[Paper](https://openaccess.thecvf.com/content/WACV2022/papers/Gudovskiy_CFLOW-AD_Real-Time_Unsupervised_Anomaly_Detection_With_Localization_via_Conditional_Normalizing_WACV_2022_paper.pdf)]
[[Code](github.com/gudovskiy/cflow-ad)]<br>
*Datasets: MVTec AD*<br>

* Multi-Scale Patch-Based Representation Learning for Image Anomaly Detection and Segmentation (WACV 2022) 
[[Paper](https://openaccess.thecvf.com/content/WACV2022/papers/Tsai_Multi-Scale_Patch-Based_Representation_Learning_for_Image_Anomaly_Detection_and_Segmentation_WACV_2022_paper.pdf)]<br>
*Datasets: MVTec AD, BTAD*<br>
<!--#### IJCV-->
#### BMVC

* Anomaly Detection and Localization Using Attention-Guided Synthetic Anomaly and Test-Time Adaptation (BMVC 2022) 
[[Paper](https://bmvc2022.mpi-inf.mpg.de/0472.pdf)]<br>
*Datasets: MVTec AD, NIH*<br>

* Siamese U-Net for Image Anomaly Detection and Segmentation with Contrastive Learning (BMVC 2022) 
[[Paper](https://bmvc2022.mpi-inf.mpg.de/0752.pdf)]<br>
*Datasets: MVTec AD,  MVTec3D-AD*<br>

* G-CMP: Graph-enhanced Contextual Matrix Profile for unsupervised anomaly detection in sensor-based remote health monitoring (BMVC 2022) 
[[Paper](https://bmvc2022.mpi-inf.mpg.de/0854.pdf)]<br>
*Datasets: Two real-world sensor-based remote health monitoringdatasets collected from the homes of persons living with dementia between August 2019 andApril 2022, by the UK Dementia Research Institute Care Research and Technology Centre*<br>

<!--#### ICCw
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
#### TPAMI-->
#### CVPRw
* Fuse-PN: A Novel Architecture for Anomaly Pattern Segmentation in Aerial Agricultural Images (CVPRw 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021W/AgriVision/papers/Innani_Fuse-PN_A_Novel_Architecture_for_Anomaly_Pattern_Segmentation_in_Aerial_CVPRW_2021_paper.pdf)]
<!--#### WACV
#### IJCV
#### BMVC
#### ICCw
#### Arxiv & Others-->
<!----------------------------------------------------------------------------------------------------------------------------------------------->
### Older Papers


* RaPP: Novelty Detection with Reconstruction along Projection Pathway (ICLR 2020) 
[[Paper](https://openreview.net/forum?id=HkgeGeBYDB)]
[[Code](https://drive.google.com/drive/folders/1sknl_i4zmvSsPYZdzYxbg66ZSYDZ_abg?usp=sharing)]<br>
*Datasets: fMNIST, MNIST, MI-F and MI-V, STL, OTTO, SNSR, EOPT, NASA, RARM*<br>
*Task: Image Classification, Anomaly Detection*

* Interpretable, Multidimensional, Multimodal Anomaly Detection with Negative Sampling for Detection of Device Failure (ICML 2020) 
[[Paper](https://arxiv.org/abs/2007.10088)]<br>
*Datasets: FOREST COVER(FC), MAMMOGRAPHY(MM), SMARTBUILDINGS(SB), MULCROSS(MC), SATELLITE(SA), SHUTTLE(SH)*<br>


* Unsupervised Ensemble-Kernel Principal Component Analysis for Hyperspectral Anomaly Detection (CVPRw 2020) 
[[Paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w6/Merrill_Unsupervised_Ensemble-Kernel_Principal_Component_Analysis_for_Hyperspectral_Anomaly_Detection_CVPRW_2020_paper.pdf)]


* Superpixel Masking and Inpainting for Self-Supervised Anomaly Detection (BMVC 2020) 
[[Paper](https://www.bmvc2020-conference.com/assets/papers/0275.pdf)]<br>
*Datasets: MVTec AD*<br>


* Deep Anomaly Detection with Outlier Exposure (ICLR 2019) 
[[Paper](https://openreview.net/forum?id=HyxCxhRcY7)]
[[Code](https://github.com/hendrycks/outlier-exposure)]<br>
*Datasets: CIFAR-10, CIFAR-100, Places, SST, SVHN, Tiny ImageNet, Tiny Images*<br>

* Anomaly Detection With Multiple-Hypotheses Predictions (ICML 2019) 
[[Paper](https://arxiv.org/abs/1810.13292)]<br>
*Datasets: CIFAR-10*<br>

* Deep Anomaly Detection for Generalized Face Anti-Spoofing (CVPRw 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/CFS/Perez-Cabo_Deep_Anomaly_Detection_for_Generalized_Face_Anti-Spoofing_CVPRW_2019_paper.pdf)]

* Anomaly-Based Manipulation Detection in Satellite Images (CVPRw 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/Media%20Forensics/Horvath_Anomaly-Based_Manipulation_Detection_in_Satellite_Images_CVPRW_2019_paper.pdf)]

* Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection (ICLR 2018) 
[[Paper](https://openreview.net/forum?id=BJJLHbb0-)]
[[Code](https://paperswithcode.com/paper/?openreview=BJJLHbb0-)]<br>
*Datasets: CIFAR-10, Fashion-MNIST, MNIST, STL-10, cats_vs_dogs*<br>


* Efficient Anomaly Detection via Matrix Sketching (NeurIPS 2018) 
[[Paper](https://papers.nips.cc/paper/2018/hash/34adeb8e3242824038aa65460a47c29e-Abstract.html)]<br>
*Datasets: p53 mutants, Dorothea and RCV1*<br>

* Deep Anomaly Detection Using Geometric Transformations (NeurIPS 2018) 
[[Paper](https://papers.nips.cc/paper/2018/hash/5e62d03aec0d17facfc5355dd90d441c-Abstract.html)]
[[Code](https://github.com/izikgo/AnomalyDetectionTransformations)]<br>
*Datasets: CIFAR-10, CIFAR-100, CatsVsDogs, fashion-MNIST*<br>

* A loss framework for calibrated anomaly detection (NeurIPS 2018) 
[[Paper](https://papers.nips.cc/paper/2018/hash/959a557f5f6beb411fd954f3f34b21c3-Abstract.html)]<br>

<!----------------------------------------------------------------------------------------------------------------------------------------------->
# Video Open World Papers
<!----------------------------------------------------------------------------------------------------------------------------------------------->

<!----------------------------------------------------------------------------------------------------------------------------------------------->
## Zero-Shot Learning Videos
<!----------------------------------------------------------------------------------------------------------------------------------------------->
### Surveys
* Zero-Shot Action Recognition in Videos: A Survey (Neurocomputing 2021) 
[[Paper](https://arxiv.org/abs/1909.06423)]<br>

* A Review of Generalized Zero-Shot Learning Methods (TPAMI 2022) 
[[Paper](https://arxiv.org/abs/2011.08641)]<br>

* Generalized Zero-Shot Learning for Action Recognition with Web-Scale Video Data (Arxiv 2017) 
[[Paper](https://arxiv.org/abs/1710.07455)]<br>

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

* Language-Free Training for Zero-Shot Video Grounding (WACV 2023) 
[[Paper](http://arxiv.org/abs/2210.12977)]<br>
*Datasets: Charades-STA, ActivityNet Captions*<br>
*Task: Video Grounding*

* Semantics Guided Contrastive Learning of Transformers for Zero-Shot Temporal Activity Detection (WACV 2023) 
[[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Nag_Semantics_Guided_Contrastive_Learning_of_Transformers_for_Zero-Shot_Temporal_Activity_WACV_2023_paper.pdf)]<br>
*Datasets: Thumos’14 and Charades*<br>
*Task: Action Recognition*
<!--#### IJCV
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
#### CVPR
* Recognizing Actions in Videos From Unseen Viewpoints (CVPR 2021) 
[[Paper](http://arxiv.org/abs/2103.16516)]<br>
*Datasets: Human3.6M, MLB-YouTube, Toyota SmartHome (TSH), NTU-RGB-D*<br>
*Task: Action Recognition*
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
#### IJCV-->
#### BMVC

* Zero-Shot Action Recognition from Diverse Object-Scene Compositions (BMVC 2021) 
[[Paper](https://www.bmvc2021-virtualconference.com/assets/papers/0739.pdf)]
[[Code](https://github.com/carlobretti/object-scene-compositions-for-actions)]<br>
*Datasets: UCF-101, Kinetics-400*<br>
*Task: Action Recognition*


<!--#### ICCw
#### Arxiv & Others-->

### Older Papers

* Out-Of-Distribution Detection for Generalized Zero-Shot Action Recognition (CVPR 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Mandal_Out-Of-Distribution_Detection_for_Generalized_Zero-Shot_Action_Recognition_CVPR_2019_paper.pdf)]
[[Code](https://github.com/naraysa/gzsl-od)]<br>
*Datasets: Olympic Sports, HMDB51 and UCF101*<br>
*Task: Action Recognition*


* Towards Universal Representation for Unseen Action Recognition (CVPR 2018) 
[[Paper](http://arxiv.org/abs/1803.08460)]<br>
*Datasets: ActivityNet, HMDB51 and UCF101*<br>
*Task: Action Recognition*
<!----------------------------------------------------------------------------------------------------------------------------------------------->
## Out-of-Distribution Detection Videos
<!----------------------------------------------------------------------------------------------------------------------------------------------->

### 2023 Papers
#### CVPR
* Towards Generalisable Video Moment Retrieval: Visual-Dynamic Injection to Image-Text Pre-Training (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.00040)]
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
<!--### 2021 Papers
 #### CVPR
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

* Out-Of-Distribution Detection for Generalized Zero-Shot Action Recognition (CVPR 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Mandal_Out-Of-Distribution_Detection_for_Generalized_Zero-Shot_Action_Recognition_CVPR_2019_paper.pdf)]
[[Code](https://github.com/naraysa/gzsl-od)]<br>
*Datasets: Olympic Sports, HMDB51 and UCF101*<br>
*Task: Action Recognition*

<!----------------------------------------------------------------------------------------------------------------------------------------------->
##  Open-Set Recognition Videos
<!----------------------------------------------------------------------------------------------------------------------------------------------->

### 2023 Papers
#### CVPR

* Enlarging Instance-specific and Class-specific Information for Open-set Action Recognition (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.15467)]

* AutoLabel: CLIP-based framework for Open-set Video Domain Adaptation (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2304.01110)]

* Open Set Action Recognition via Multi-Label Evidential Learning (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.12698)]

* OpenGait: Revisiting Gait Recognition Towards Better Practicality (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2211.06597)]

* Open-Category Human-Object Interaction Pre-training via Language Modeling Framework (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Zheng_Open-Category_Human-Object_Interaction_Pre-Training_via_Language_Modeling_Framework_CVPR_2023_paper.html)]

* SUDS: Scalable Urban Dynamic Scenes (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.14536)]

<!-- #### ICLR
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
#### ICCw-->
#### Arxiv & Others

* Video Instance Segmentation in an Open-World (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2304.01200)]

* Plan4MC: Skill Reinforcement Learning and Planning for Open-World Minecraft Tasks (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.16563)]

* POAR: Towards Open-World Pedestrian Attribute Recognition (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.14643)]

* Learning to Operate in Open Worlds by Adapting Planning Models (AAMAS 2023) 
[[Paper](https://arxiv.org/abs/2303.14272)]

* PyReason: Software for Open World Temporal Logic (AAAI 2023) 
[[Paper](https://arxiv.org/abs/2302.13482)]

* NovPhy: A Testbed for Physical Reasoning in Open-world Environments (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.01711)]

* Improving Audio-Visual Video Parsing with Pseudo Visual Labels (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.02344)]

* Open-World Object Manipulation using Pre-trained Vision-Language Models (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.00905)]

* Towards Generalized Robot Assembly through Compliance-Enabled Contact Formations (ICRA 2023) 
[[Paper](https://arxiv.org/abs/2303.05565)]

* Discovering Novel Actions in an Open World with Object-Grounded Visual Commonsense Reasoning (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2305.16602)]

* Temporal-controlled Frame Swap for Generating High-Fidelity Stereo Driving Data for Autonomy Analysis (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2306.01704)]


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
#### ICCw-->
#### Arxiv & Others
* Human Activity Recognition in an Open World (Submitted to JAIR 2022) 
[[Paper](https://arxiv.org/abs/2212.12141)]

* Self-Initiated Open World Learning for Autonomous AI Agents (AAAI 2022 Spring Symposium Series) 
[[Paper](https://arxiv.org/abs/2110.11385)] 

* UVO Challenge on Video-based Open-World Segmentation 2021: 1st Place Solution (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2110.11661)] 

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

* Conditional Extreme Value Theory for Open Set Video Domain Adaptation (MMAsia 2021) 
[[Paper](https://dl.acm.org/doi/10.1145/3469877.3490600)]<br>

* Dual Metric Discriminator for Open Set Video Domain Adaptation (ICASSP 2021) 
[[Paper](https://ieeexplore.ieee.org/document/9413361)]<br> 

* Open-World Active Learning with Stacking Ensemble for Self-Driving Cars (Arxiv 2021) 
[[Paper](https://arxiv.org/abs/2109.06628)] 

* Physical Reasoning in an Open World (ACS 2021) 
[[Paper](https://arxiv.org/abs/2201.08950)] 

* Person Re-identification based on Robust Features in Open-world (Arxiv 2021) 
[[Paper](https://arxiv.org/abs/2102.10798)] 

* Online Action Recognition (AAAI 2021) 
[[Paper](https://arxiv.org/abs/2012.07464)] 

* ViNG: Learning Open-World Navigation with Visual Goals (ICRA 2021) 
[[Paper](https://arxiv.org/abs/2012.09812)] 

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

* Open-World Stereo Video Matching with Deep RNN (ECCV 2018) 
[[Paper](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Yiran_Zhong_Open-World_Stereo_Video_ECCV_2018_paper.pdf)]<br>
*Datasets: KITTI VO, Middlebury Stereo 2005 & 2006, Freiburg Sceneflow, Random dot, Synthia*<br>
*Task: Stereo Video Matching*

* Adversarial Open-World Person Re-Identification (ECCV 2018) 
[[Paper](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Xiang_Li_Adversarial_Open-World_Person_ECCV_2018_paper.pdf)]<br>
*Datasets: Market-1501, CUHK01, CUHK03*<br>
*Task: Person Re-Identification*

* From Open Set to Closed Set: Counting Objects by Spatial Divide-and-Conquer (ICCV 2019) 
[[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Xiong_From_Open_Set_to_Closed_Set_Counting_Objects_by_Spatial_ICCV_2019_paper.pdf)]
[[Code](https://github.com/xhp-hust-2018-2011/S-DCNet)]<br>
*Datasets: Synthesized Cell Counting, UCF-QNRF, ShanghaiTech, UCFCC50, TRANCOS and MTC*<br>
*Task: Visual Counting*

* AutOTranS: an Autonomous Open World Transportation System (Arxiv 2018) 
[[Paper](https://arxiv.org/abs/1810.03400)] 

* From Known to the Unknown: Transferring Knowledge to Answer Questions about Novel Visual and Semantic Concepts (Arxiv 2018) 
[[Paper](https://arxiv.org/abs/1811.12772)] 

* Visual Curiosity: Learning to Ask Questions to Learn Visual Recognition (CoRL 2018 Oral) 
[[Paper](https://arxiv.org/abs/1810.00912)] 

* Towards Large-Scale Video Video Object Mining (ECCVw 2018) 
[[Paper](https://arxiv.org/abs/1809.07316)] 

<!----------------------------------------------------------------------------------------------------------------------------------------------->
##  Novel Class Discovery Videos
<!----------------------------------------------------------------------------------------------------------------------------------------------->

### 2023 Papers
#### CVPR

* Open-Category Human-Object Interaction Pre-training via Language Modeling Framework (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Zheng_Open-Category_Human-Object_Interaction_Pre-Training_via_Language_Modeling_Framework_CVPR_2023_paper.html)]
<!-- #### ICLR
#### NeurIPS
#### ICCV
#### ICML

* AnoDDPM: Anomaly Detection With Denoising Diffusion Probabilistic Models Using Simplex Noise (CVPRw 2022) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Wyatt_AnoDDPM_Anomaly_Detection_With_Denoising_Diffusion_Probabilistic_Models_Using_Simplex_CVPRW_2022_paper.pdf)]
[[Code](https://github.com/Julian-Wyatt/AnoDDPM)]<br>
*Datasets: MVTec AD*<br>
#### IEEE-Access
#### ECCV
#### AAAI
#### TPAMI
#### CVPRw
#### WACV
#### IJCV
#### BMVC
#### ICCw-->
#### Arxiv & Others

* NEV-NCD: Negative Learning, Entropy, and Variance regularization based novel action categories discovery (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2304.07354)]
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
#### IJCV-->
#### BMVC

* Unsupervised Discovery of Actions in Instructional Videos (BMVC 2021) 
[[Paper](https://www.bmvc2021-virtualconference.com/assets/papers/0773.pdf)]<br>
*Datasets: 50-salads dataset, Narrated Instructional Videos (NIV) dataset, Breakfast dataset*<br>
*Task: Action Discovery*

<!--#### ICCw
#### Arxiv & Others-->
<!----------------------------------------------------------------------------------------------------------------------------------------------->
### Older Papers

* Tracking the Known and the Unknown by Leveraging Semantic Information (BMVC 2019) 
[[Paper](https://bmvc2019.org/wp-content/papers/1003.html)]
[[Code](https://tracking.vision.ee.ethz.ch/track-known-unknown/)]<br>
*Datasets: NFS, UAV123, LaSOT, TrackingNet, VOT2018*<br>
*Task: Object Tracking*

* DetectFusion: Detecting and Segmenting Both Known and Unknown Dynamic Objects in Real-time SLAM (BMVC 2019) 
[[Paper](https://bmvc2019.org/wp-content/papers/0499.html)]<br>
*Datasets: TUM-RGB-D, MS COCO, PASCAL VOC*<br>
*Task: Object Tracking and Segmentation*

* Localizing Novel Attended Objects in Egocentric Views (BMVC 2020) 
[[Paper](https://www.bmvc2020-conference.com/assets/papers/0014.pdf)]<br>
*Datasets: GTEA Gaze+, Toy Room*<br>
*Task: Novel Object Localization*

* Video Face Clustering With Unknown Number of Clusters (ICCV 2019) 
[[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Tapaswi_Video_Face_Clustering_With_Unknown_Number_of_Clusters_ICCV_2019_paper.pdf)]
[[Code](https://github.com/makarandtapaswi/BallClustering_ICCV2019)]<br>
*Datasets: MovieGraphs, The Big Bang Theory (BBT) and Buffy the Vampire Slayer (BUFFY)*<br>
*Task: Face Clustering*

* Incremental Class Discovery for Semantic Segmentation With RGBD Sensing (ICCV 2019) 
[[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Nakajima_Incremental_Class_Discovery_for_Semantic_Segmentation_With_RGBD_Sensing_ICCV_2019_paper.pdf)]<br>
*Datasets: NYUDv2*<br>
*Task: Semantic Segmentation*

* Object Discovery in Videos as Foreground Motion Clustering (ICCV 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xie_Object_Discovery_in_Videos_as_Foreground_Motion_Clustering_CVPR_2019_paper.pdf)]<br>
*Datasets: Flying Things 3d (FT3D), DAVIS2016, Freibug-Berkeley motion segmentation, Complex Background, and Camouflaged Animal*<br>
*Task: Object Discovery*


<!----------------------------------------------------------------------------------------------------------------------------------------------->
## Open Vocabulary Videos
<!----------------------------------------------------------------------------------------------------------------------------------------------->

### 2023 Papers
#### CVPR

* Open-Category Human-Object Interaction Pre-training via Language Modeling Framework (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Zheng_Open-Category_Human-Object_Interaction_Pre-Training_via_Language_Modeling_Framework_CVPR_2023_paper.html)]

* Being Comes from Not-being: Open-vocabulary Text-to-Motion Generation with Wordless Training (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2210.15929)]

* OVTrack: Open-Vocabulary Multiple Object Tracking (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Li_OVTrack_Open-Vocabulary_Multiple_Object_Tracking_CVPR_2023_paper.html)]
#### ICLR
* The Devil is in the Wrongly-classified Samples: Towards Unified Open-set Recognition (ICLR 2023) 
[[Paper](https://openreview.net/pdf?id=xLr0I_xYGAs)]<br>
*Datasets: CIFAR100, LSUN, MiTv2, UCF101, HMDB51*<br>
*Task: Image and Video Classification*
<!--#### NeurIPS
#### ICCV-->
#### ICML

* Open-VCLIP: Transforming CLIP to an Open-vocabulary Video Model via Interpolated Weight Optimization (ICML 2023) 
[[Paper](https://arxiv.org/abs/2302.00624)]

<!--#### IEEE-Access
#### ECCV
#### AAAI
#### TPAMI
#### CVPRw
#### WACV
#### IJCV
#### BMVC
#### ICCw-->
#### Arxiv & Others

* Transforming CLIP to an Open-vocabulary Video Model via Interpolated Weight Optimization (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2302.00624)]

* TagCLIP: Improving Discrimination Ability of Open-Vocabulary Semantic Segmentation (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2304.07547)]

* MVP-SEG: Multi-View Prompt Learning for Open-Vocabulary Semantic Segmentation (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2304.06957)]

* Segment Everything Everywhere All at Once (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2304.06718)]

* Towards Open-Vocabulary Video Instance Segmentation (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2304.01715)]

* CLIP Surgery for Better Explainability with Enhancement in Open-Vocabulary Tasks (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2304.05653)]

* Prompt Pre-Training with Twenty-Thousand Classes for Open-Vocabulary Visual Recognition (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2304.04704)]

* V3Det: Vast Vocabulary Visual Detection Dataset (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2304.03752)]

* Token Merging for Fast Stable Diffusion (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.17604)]

* Going Beyond Nouns With Vision & Language Models Using Synthetic Data (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.17590)]

* MaMMUT: A Simple Architecture for Joint Learning for MultiModal Tasks (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.16839)]

* ZBS: Zero-shot Background Subtraction via Instance-level Background Modeling and Foreground Selection (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.14679)]

* Prompt-Guided Transformers for End-to-End Open-Vocabulary Object Detection (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.14386)]

* Three ways to improve feature alignment for open vocabulary detection (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.13518)]

* Zero-guidance Segmentation Using Zero Segment Labels (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.13396)]

* Open-Vocabulary Object Detection using Pseudo Caption Labels (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.13040)]

* Uni-Fusion: Universal Continuous Mapping (Arxiv 2023) 
[[Paper](https://arxiv.org/abs/2303.12678)]
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
#### ICCw-->
#### Arxiv & Others

* Open-Vocabulary Temporal Action Detection with Off-the-Shelf Image-Text Features (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2212.10596)]

<!----------------------------------------------------------------------------------------------------------------------------------------------->
<!--### 2021 Papers
 #### CVPR
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
#### CVPR
* On the Difficulty of Unpaired Infrared-to-Visible Video Translation: Fine-Grained Content-Rich Patches Transfer (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_On_the_Difficulty_of_Unpaired_Infrared-to-Visible_Video_Translation_Fine-Grained_Content-Rich_CVPR_2023_paper.html)]

* ANetQA: A Large-scale Benchmark for Fine-grained Compositional Reasoning over Untrimmed Videos (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_ANetQA_A_Large-Scale_Benchmark_for_Fine-Grained_Compositional_Reasoning_Over_Untrimmed_CVPR_2023_paper.html)]

* MELTR: Meta Loss Transformer for Learning to Fine-tune Video Foundation Models (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.13009)]

* Modeling Video as Stochastic Processes for Fine-Grained Video Representation Learning (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Modeling_Video_As_Stochastic_Processes_for_Fine-Grained_Video_Representation_Learning_CVPR_2023_paper.html)]

* Class Prototypes based Contrastive Learning for Classifying Multi-Label and Fine-Grained Educational Videos (CVPR 2023) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Gupta_Class_Prototypes_Based_Contrastive_Learning_for_Classifying_Multi-Label_and_Fine-Grained_CVPR_2023_paper.html)]

* Progressive Disentangled Representation Learning for Fine-Grained Controllable Talking Head Synthesis (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2211.14506)]

* Focus On Details: Online Multi-object Tracking with Diverse Fine-grained Representation (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2302.14589)]

* Fine-grained Audible Video Description (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.15616)]
<!-- #### ICLR
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
#### ICCw-->
#### Arxiv & Others
* Hand Guided High Resolution Feature Enhancement for Fine-Grained Atomic Action Segmentation Within Complex Human Assemblies (WACVw 2023) 
[[Paper](http://arxiv.org/abs/2211.13694)]

* A Transformer-Based Late-Fusion Mechanism for Fine-Grained Object Recognition in Videos (WACVw 2023) 
[[Paper](https://openaccess.thecvf.com/content/WACV2023W/RWS/papers/Koch_A_Transformer-Based_Late-Fusion_Mechanism_for_Fine-Grained_Object_Recognition_in_Videos_WACVW_2023_paper.pdf)]

* Simplifying Open-Set Video Domain Adaptation with Contrastive Learning (CVIU 2023 under review) 
[[Paper](https://arxiv.org/abs/2301.03322)]
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
*Datasets: VATEX-EVAL; ActivityNet-FOIL*<br>
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
#### CVPR

* Fine-Grained Shape-Appearance Mutual Learning for Cloth-Changing Person Re-Identification (CVPR 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Hong_Fine-Grained_Shape-Appearance_Mutual_Learning_for_Cloth-Changing_Person_Re-Identification_CVPR_2021_paper.pdf)]


* Temporal Query Networks for Fine-Grained Video Understanding (CVPR 2021) 
[[Paper](http://arxiv.org/abs/2104.09496)]

* GLAVNet: Global-Local Audio-Visual Cues for Fine-Grained Material Recognition (CVPR 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Shi_GLAVNet_Global-Local_Audio-Visual_Cues_for_Fine-Grained_Material_Recognition_CVPR_2021_paper.pdf)]


<!--#### ICLR
#### NeurIPS-->
#### ICCV

* Counterfactual Attention Learning for Fine-Grained Visual Categorization and Re-Identification (ICCV 2021) 
[[Paper](http://arxiv.org/abs/2108.08728)]

* Video Pose Distillation for Few-Shot, Fine-Grained Sports Action Recognition (ICCV 2021) 
[[Paper](http://arxiv.org/abs/2109.01305)]

* FuseFormer: Fusing Fine-Grained Information in Transformers for Video Inpainting (ICCV 2021) 
[[Paper](http://arxiv.org/abs/2109.02974)]

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
### Older Papers
* FineGym: A Hierarchical Video Dataset for Fine-Grained Action Understanding (CVPR 2020) 
[[Paper](http://arxiv.org/abs/2004.06704)]

* Multi-Modal Domain Adaptation for Fine-Grained Action Recognition (CVPR 2020) 
[[Paper](http://arxiv.org/abs/2001.09691)]

* Revisiting Pose-Normalization for Fine-Grained Few-Shot Recognition (CVPR 2020) 
[[Paper](http://arxiv.org/abs/2004.00705)]

* Fine-Grained Video-Text Retrieval With Hierarchical Graph Reasoning (CVPR 2020) 
[[Paper](http://arxiv.org/abs/2003.00392)]

* Stochastic Fine-grained Labeling of Multi-state Sign Glosses for Continuous Sign Language Recognition (ECCV 2020) 
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610171.pdf)]

* Fine-Grained Motion Representation For Template-Free Visual Tracking (WACV 2020) 
[[Paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Shuang_Fine-Grained_Motion_Representation_For_Template-Free_Visual_Tracking_WACV_2020_paper.pdf)]

* WHENet: Real-time Fine-Grained Estimation for Wide Range Head Pose (BMVC 2020) 
[[Paper](https://www.bmvc2020-conference.com/assets/papers/0907.pdf)]

* Yoga-82: A New Dataset for Fine-Grained Classification of Human Poses (CVPRw 2020) 
[[Paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w70/Verma_Yoga-82_A_New_Dataset_for_Fine-Grained_Classification_of_Human_Poses_CVPRW_2020_paper.pdf)]

* Fine-Grained Pointing Recognition for Natural Drone Guidance (CVPRw 2020) 
[[Paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w70/Barbed_Fine-Grained_Pointing_Recognition_for_Natural_Drone_Guidance_CVPRW_2020_paper.pdf)]

* Local Temporal Bilinear Pooling for Fine-Grained Action Parsing (CVPR 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Local_Temporal_Bilinear_Pooling_for_Fine-Grained_Action_Parsing_CVPR_2019_paper.pdf)]

* Drive&Act: A Multi-Modal Dataset for Fine-Grained Driver Behavior Recognition in Autonomous Vehicles (ICCV 2019) 
[[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Martin_DriveAct_A_Multi-Modal_Dataset_for_Fine-Grained_Driver_Behavior_Recognition_in_ICCV_2019_paper.pdf)]

* Fine-Grained Action Retrieval Through Multiple Parts-of-Speech Embeddings (ICCV 2019) 
[[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wray_Fine-Grained_Action_Retrieval_Through_Multiple_Parts-of-Speech_Embeddings_ICCV_2019_paper.pdf)]

* Learning Motion in Feature Space: Locally-Consistent Deformable Convolution Networks for Fine-Grained Action Detection (ICCV 2019) 
[[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Mac_Learning_Motion_in_Feature_Space_Locally-Consistent_Deformable_Convolution_Networks_for_ICCV_2019_paper.pdf)]

* ViSiL: Fine-Grained Spatio-Temporal Video Similarity Learning (ICCV 2019) 
[[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kordopatis-Zilos_ViSiL_Fine-Grained_Spatio-Temporal_Video_Similarity_Learning_ICCV_2019_paper.pdf)]

* Fine-Grained Visual Dribbling Style Analysis for Soccer Videos With Augmented Dribble Energy Image (CVPRw 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/CVSports/Li_Fine-Grained_Visual_Dribbling_Style_Analysis_for_Soccer_Videos_With_Augmented_CVPRW_2019_paper.pdf)]

* Anticipation of Human Actions With Pose-Based Fine-Grained Representations (CVPRw 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/Precognition/Agethen_Anticipation_of_Human_Actions_With_Pose-Based_Fine-Grained_Representations_CVPRW_2019_paper.pdf)]

* Fine-Grained Video Captioning for Sports Narrative (CVPR 2018) 
[[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_Fine-Grained_Video_Captioning_CVPR_2018_paper.pdf)]

* Where Will They Go? Predicting Fine-Grained Adversarial Multi-Agent Motion using Conditional Variational Autoencoders (ECCV 2018) 
[[Paper](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Panna_Felsen_Where_Will_They_ECCV_2018_paper.pdf)]

* Fine-grained Video Categorization with Redundancy Reduction Attention (ECCV 2018) 
[[Paper](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Chen_Zhu_Fine-grained_Video_Categorization_ECCV_2018_paper.pdf)]

* Fine-Grained Head Pose Estimation Without Keypoints (CVPRw 2018) 
[[Paper](http://arxiv.org/abs/1710.00925)]

* Fine-Grained Activity Recognition in Baseball Videos (CVPRw 2018) 
[[Paper](http://arxiv.org/abs/1804.03247)]

<!----------------------------------------------------------------------------------------------------------------------------------------------->
## Long Tail Videos
<!----------------------------------------------------------------------------------------------------------------------------------------------->

### 2023 Papers
#### CVPR
* Use Your Head: Improving Long-Tail Video Recognition (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2304.01143)]

* FEND: A Future Enhanced Distribution-Aware Contrastive Learning Framework For Long-tail Trajectory Prediction (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.16574)]

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
<!--### 2022 Papers
#### CVPR
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
<!--#### CVPR
#### ICLR
#### NeurIPS-->
#### ICCV
* VideoLT: Large-Scale Long-Tailed Video Recognition (ICCV 2021) 
[[Paper](http://arxiv.org/abs/2105.02668)]

* On Exposing the Challenging Long Tail in Future Prediction of Traffic Actors (ICCV 2021) 
[[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Makansi_On_Exposing_the_Challenging_Long_Tail_in_Future_Prediction_of_ICCV_2021_paper.pdf)]
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
## Anomaly Detection Videos
<!----------------------------------------------------------------------------------------------------------------------------------------------->
### 2023 Papers
<!-- #### CVPR
#### ICLR
#### NeurIPS-->
#### ICCV
* Multimodal Motion Conditioned Diffusion Model for Skeleton-based Video Anomaly Detection (ICCV 2023) 
[[Paper](https://arxiv.org/abs/2307.07205)]<br>

<!--#### ICML
#### IEEE-Access
#### ECCV
#### AAAI
#### TPAMI
#### CVPRw-->
#### WACV
* DyAnNet: A Scene Dynamicity Guided Self-Trained Video Anomaly Detection Network (WACV 2023) 
[[Paper](http://arxiv.org/abs/2211.00882)]<br>
*Datasets: UCF-Crime, CCTV-Fights, UBI-Fights*

* Cross-Domain Video Anomaly Detection Without Target Domain Adaptation (WACV 2023) 
[[Paper](http://arxiv.org/abs/2212.07010)]<br>
*Datasets: SHTdc, SHT and Ped2, HMDB, UCF101*

* Bi-Directional Frame Interpolation for Unsupervised Video Anomaly Detection (WACV 2023) 
[[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Deng_Bi-Directional_Frame_Interpolation_for_Unsupervised_Video_Anomaly_Detection_WACV_2023_paper.pdf)]<br>
*Datasets: UCSD Ped2, CUHK Avenue, ShanghaiTech Campus*

* Towards Interpretable Video Anomaly Detection (WACV 2023) 
[[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Doshi_Towards_Interpretable_Video_Anomaly_Detection_WACV_2023_paper.pdf)]<br>
*Datasets: CUHK Avenue, ShanghaiTech Campus*

* Normality Guided Multiple Instance Learning for Weakly Supervised Video Anomaly Detection (WACV 2023) 
[[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Park_Normality_Guided_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_WACV_2023_paper.pdf)]<br>
*Datasets: ShanghaiTech, UCF-Crime, XD-Violence*


<!--#### IJCV
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
#### ICCV-->
#### ICML

* Latent Outlier Exposure for Anomaly Detection with Contaminated Data (ICML 2022) 
[[Paper](https://arxiv.org/abs/2202.08088)]
[[Code](https://github.com/boschresearch/LatentOE-AD.git)]<br>
*Datasets:  CIFAR-10, Fashion-MNIST, MVTEC, 30 tabular data sets, UCSD Peds1*<br>

<!--#### IEEE-Access-->
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

#### WACV
* A Modular and Unified Framework for Detecting and Localizing Video Anomalies (WACV 2022) 
[[Paper](http://arxiv.org/abs/2103.11299)]<br>
*Datasets: CUHK Avenue, UCSD Ped2, ShanghaiTech Campus, UR fall*<br>

* FastAno: Fast Anomaly Detection via Spatio-Temporal Patch Transformation (WACV 2022) 
[[Paper](http://arxiv.org/abs/2106.08613)]<br>
*Datasets: CUHK Avenue, UCSD Ped2, ShanghaiTech Campus*<br>

* Multi-Branch Neural Networks for Video Anomaly Detection in Adverse Lighting and Weather Conditions (WACV 2022) 
[[Paper](https://openaccess.thecvf.com/content/WACV2022/papers/Leroux_Multi-Branch_Neural_Networks_for_Video_Anomaly_Detection_in_Adverse_Lighting_WACV_2022_paper.pdf)]
[[Code](https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library)]<br>
*Datasets: CUHK Avenue (Augmented)*<br>

* Discrete Neural Representations for Explainable Anomaly Detection (WACV 2022) 
[[Paper](http://arxiv.org/abs/2112.05585)]
[[Code](http://jjcvision.com/projects/vqunet_anomally_detection.html)]<br>
*Datasets: CUHK Avenue, UCSD Ped2, X-MAN*<br>

* Rethinking Video Anomaly Detection - A Continual Learning Approach (WACV 2022) 
[[Paper](https://openaccess.thecvf.com/content/WACV2022/papers/Doshi_Rethinking_Video_Anomaly_Detection_-_A_Continual_Learning_Approach_WACV_2022_paper.pdf)]<br>
*Datasets: NOLA*<br>


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
#### TPAMI-->
#### CVPRw
* Box-Level Tube Tracking and Refinement for Vehicles Anomaly Detection (CVPRw 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021W/AICity/papers/Wu_Box-Level_Tube_Tracking_and_Refinement_for_Vehicles_Anomaly_Detection_CVPRW_2021_paper.pdf)]

* Dual-Modality Vehicle Anomaly Detection via Bilateral Trajectory Tracing (CVPRw 2021) 
[[Paper](http://arxiv.org/abs/2106.05003)]

* A Vision-Based System for Traffic Anomaly Detection Using Deep Learning and Decision Trees (CVPRw 2021) 
[[Paper](http://arxiv.org/abs/2104.06856)]

* Good Practices and a Strong Baseline for Traffic Anomaly Detection (CVPRw 2021) 
[[Paper](http://arxiv.org/abs/2105.03827)]

* An Efficient Approach for Anomaly Detection in Traffic Videos (CVPRw 2021) 
[[Paper](http://arxiv.org/abs/2104.09758)]


* Spacecraft Time-Series Anomaly Detection Using Transfer Learning (CVPRw 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021W/AI4Space/papers/Baireddy_Spacecraft_Time-Series_Anomaly_Detection_Using_Transfer_Learning_CVPRW_2021_paper.pdf)]
<!--#### WACV
#### IJCV
#### BMVC
#### ICCw
#### Arxiv & Others-->

<!----------------------------------------------------------------------------------------------------------------------------------------------->
### Older Papers

* Multi-Granularity Tracking With Modularlized Components for Unsupervised Vehicles Anomaly Detection (CVPRw 2020) 
[[Paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w35/Li_Multi-Granularity_Tracking_With_Modularlized_Components_for_Unsupervised_Vehicles_Anomaly_Detection_CVPRW_2020_paper.pdf)]

* Fractional Data Distillation Model for Anomaly Detection in Traffic Videos (CVPRw 2020) 
[[Paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w35/Shine_Fractional_Data_Distillation_Model_for_Anomaly_Detection_in_Traffic_Videos_CVPRW_2020_paper.pdf)]

* Towards Real-Time Systems for Vehicle Re-Identification, Multi-Camera Tracking, and Anomaly Detection (CVPRw 2020) 
[[Paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w35/Peri_Towards_Real-Time_Systems_for_Vehicle_Re-Identification_Multi-Camera_Tracking_and_Anomaly_CVPRW_2020_paper.pdf)]

* Fast Unsupervised Anomaly Detection in Traffic Videos (CVPRw 2020) 
[[Paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w35/Doshi_Fast_Unsupervised_Anomaly_Detection_in_Traffic_Videos_CVPRW_2020_paper.pdf)]

* Continual Learning for Anomaly Detection in Surveillance Videos (CVPRw 2020) 
[[Paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w15/Doshi_Continual_Learning_for_Anomaly_Detection_in_Surveillance_Videos_CVPRW_2020_paper.pdf)]

* Any-Shot Sequential Anomaly Detection in Surveillance Videos (CVPRw 2020) 
[[Paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w54/Doshi_Any-Shot_Sequential_Anomaly_Detection_in_Surveillance_Videos_CVPRW_2020_paper.pdf)]


* Challenges in Time-Stamp Aware Anomaly Detection in Traffic Videos (CVPRw 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/AI%20City/Biradar_Challenges_in_Time-Stamp_Aware_Anomaly_Detection_in_Traffic_Videos_CVPRW_2019_paper.pdf)]

* Traffic Anomaly Detection via Perspective Map based on Spatial-temporal Information Matrix (CVPRw 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/AI%20City/Bai_Traffic_Anomaly_Detection_via_Perspective_Map_based_on_Spatial-temporal_Information_CVPRW_2019_paper.pdf)]

* Unsupervised Traffic Anomaly Detection Using Trajectories (CVPRw 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/AI%20City/Zhao_Unsupervised_Traffic_Anomaly_Detection_Using_Trajectories_CVPRW_2019_paper.pdf)]

* Attention Driven Vehicle Re-identification and Unsupervised Anomaly Detection for Traffic Understanding (CVPRw 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/AI%20City/Khorramshahi_Attention_Driven_Vehicle_Re-identification_and_Unsupervised_Anomaly_Detection_for_Traffic_CVPRW_2019_paper.pdf)]

* A Comparative Study of Faster R-CNN Models for Anomaly Detection in 2019 AI City Challenge (CVPRw 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/AI%20City/Shine_A_Comparative_Study_of_Faster_R-CNN_Models_for_Anomaly_Detection_CVPRW_2019_paper.pdf)]

* Anomaly Candidate Identification and Starting Time Estimation of Vehicles from Traffic Videos (CVPRw 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/AI%20City/Wang_Anomaly_Candidate_Identification_and_Starting_Time_Estimation_of_Vehicles_from_CVPRW_2019_paper.pdf)]

* Hybrid Deep Network for Anomaly Detection (BMVC 2019) 
[[Paper](https://bmvc2019.org/wp-content/papers/0726.html)]<br>
*Datasets: CUHK Avenue, UCSD Ped2, Belleview, Traffic-Train*<br>

* Motion-Aware Feature for Improved Video Anomaly Detection (BMVC 2019) 
[[Paper](https://bmvc2019.org/wp-content/papers/0129.html)]<br>
*Datasets: UCF Crime*<br>

* Adversarially Learned One-Class Classifier for Novelty Detection (CVPR 2018) 
[[Paper](http://arxiv.org/abs/1802.09088)]<br>
*Datasets: MNIST, Caltech-256, UCSD Ped2*<br>
*Task: Image Classification, Anomaly Detection*

* Real-World Anomaly Detection in Surveillance Videos (CVPR 2018) 
[[Paper](http://arxiv.org/abs/1801.04264)]
[[Code](http://crcv.ucf.edu/projects/real-world/)]<br>
*Datasets: Real-world Surveillance Videos*<br>

* Future Frame Prediction for Anomaly Detection – A New Baseline (CVPR 2018) 
[[Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Liu_Future_Frame_Prediction_CVPR_2018_paper.html)]
[[Code](https://github.com/StevenLiuWen/ano_pred_cvpr2018)]<br>
*Datasets: CUHK, Avenue, UCSD Ped1, UCSD Ped2, ShanghaiTech, Paper's toy dataset*<br>

* Unsupervised Anomaly Detection for Traffic Surveillance Based on Background Modeling (CVPRw 2018) 
[[Paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w3/Wei_Unsupervised_Anomaly_Detection_CVPR_2018_paper.pdf)]

* Dual-Mode Vehicle Motion Pattern Learning for High Performance Road Traffic Anomaly Detection (CVPRw 2018) 
[[Paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w3/Xu_Dual-Mode_Vehicle_Motion_CVPR_2018_paper.pdf)]

<!----------------------------------------------------------------------------------------------------------------------------------------------->
## Novelty Detection
<!----------------------------------------------------------------------------------------------------------------------------------------------->
<!--### 2023 Papers
#### CVPR
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
<!--### 2022 Papers
#### CVPR
#### ICLR
#### NeurIPS
#### ICCV
#### ICML
#### IEEE-Access-->
#### ECCV
* incDFM: Incremental Deep Feature Modeling for Continual Novelty Detection (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850581.pdf)]<br>
*Datasets: 1.CIFAR-10 (10 classes), 2. CIFAR-100 (super-classlevel, 20 classes), 3. EMNIST (26 classes) and 4. iNaturalist21 (phylumlevel, 9 classes)*<br>
*Task: Image Classification*
<!--#### AAAI
#### TPAMI
#### CVPRw-->
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
#### CVPR
* Learning Deep Classifiers Consistent With Fine-Grained Novelty Detection (CVPR 2021) 
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Cheng_Learning_Deep_Classifiers_Consistent_With_Fine-Grained_Novelty_Detection_CVPR_2021_paper.pdf)]<br>
*Datasets: small- and large-scale FGVC*<br>
*Task: Novelty Detection*

<!--#### ICLR
#### NeurIPS
#### ICCV
#### ICML
#### IEEE-Access
#### ECCV-->
#### AAAI

* A Unifying Framework for Formal Theories of Novelty:Framework, Examples and Discussion (AAAI 2021) 
[[Paper](https://arxiv.org/abs/2012.06957)] 

<!--#### TPAMI
#### CVPRw
#### WACV
#### IJCV-->
#### BMVC

* Multi-Class Novelty Detection with Generated Hard Novel Features (BMVC 2021) 
[[Paper](https://www.bmvc2021-virtualconference.com/assets/papers/0838.pdf)]<br>
*Datasets: Stanford Dogs, Caltech 256, CUB 200, FounderType-200*<br>
*Task: Image Classification*

<!--#### ICCw
#### Arxiv & Others-->

<!----------------------------------------------------------------------------------------------------------------------------------------------->
### Older Papers
<!----------------------------------------------------------------------------------------------------------------------------------------------->
* Improving Exploration in Evolution Strategies for Deep Reinforcement Learning via a Population of Novelty-Seeking Agents (NeurIPS 2018) 
[[Paper](https://papers.nips.cc/paper/2018/hash/b1301141feffabac455e1f90a7de2054-Abstract.html)]<br>
*Datasets: OpenAI Gym*<br>
*Task: Reinforcement Learning*

* Multivariate Triangular Quantile Maps for Novelty Detection (NeurIPS 2019) 
[[Paper](https://papers.nips.cc/paper/2019/hash/6244b2ba957c48bc64582cf2bcec3d04-Abstract.html)]
[[Code](https://github.com/GinGinWang/MTQ)]<br>
*Datasets: MNIST and Fashion-MNIST, KDDCUP and Thyroid*<br>
*Task: Image Classification*

* Multi-class Novelty Detection Using Mix-up Technique (WACV 2020) 
[[Paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Bhattacharjee_Multi-class_Novelty_Detection_Using_Mix-up_Technique_WACV_2020_paper.pdf)]<br>
*Datasets: Caltech 256 and Stanford Dogs*<br>
*Task: Image Classification*

* Hierarchical Novelty Detection for Visual Object Recognition (CVPR 2018) 
[[Paper](http://arxiv.org/abs/1804.00722)]<br>
*Datasets: ImageNet, AwA2, CUB*<br>
*Task: Image Classification*

* Adversarially Learned One-Class Classifier for Novelty Detection (CVPR 2018) 
[[Paper](http://arxiv.org/abs/1802.09088)]<br>
*Datasets: MNIST, Caltech-256, UCSD Ped2*<br>
*Task: Image Classification, Anomaly Detection*

* Multiple Class Novelty Detection Under Data Distribution Shift (ECCV 2020) 
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520426.pdf)]<br>
*Datasets: SVHN, MNIST and USPS, Office-31*<br>
*Task: Image Classification*

* Utilizing Patch-level Category Activation Patterns for Multiple Class Novelty Detection (ECCV 2020) 
[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550426.pdf)]<br>
*Datasets: Caltech256, CUB-200, Stanford Dogs and FounderType-200*<br>
*Task: Image Classification*

* Unsupervised and Semi-supervised Novelty Detection using Variational Autoencoders in Opportunistic Science Missions (BMVC 2020) 
[[Paper](https://www.bmvc2020-conference.com/assets/papers/0643.pdf)]<br>
*Datasets: Mars novelty detection Mastcam labeled dataset*<br>
*Task: Image Classification*

* Where's Wally Now? Deep Generative and Discriminative Embeddings for Novelty Detection (CVPR 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Burlina_Wheres_Wally_Now_Deep_Generative_and_Discriminative_Embeddings_for_Novelty_CVPR_2019_paper.pdf)]<br>
*Datasets: CIFAR-10, IN-125*<br>
*Task: Image Classification*

* Deep Transfer Learning for Multiple Class Novelty Detection (CVPR 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Perera_Deep_Transfer_Learning_for_Multiple_Class_Novelty_Detection_CVPR_2019_paper.pdf)]<br>
*Datasets: Caltech256, Caltech-UCSD Birds 200 (CUB 200), Stanford Dogs, FounderType-200*<br>
*Task: Image Classification*


* Latent Space Autoregression for Novelty Detection (CVPR 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Abati_Latent_Space_Autoregression_for_Novelty_Detection_CVPR_2019_paper.pdf)]
[[Code](https://github.com/aimagelab/novelty-detection)]<br>
*Datasets: MNIST, CIFAR10, UCSD Ped2 and ShanghaiTech*<br>
*Task: Image Classification, Video Anomaly Detection*


* OCGAN: One-Class Novelty Detection Using GANs With Constrained Latent Representations (CVPR 2019) 
[[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Perera_OCGAN_One-Class_Novelty_Detection_Using_GANs_With_Constrained_Latent_Representations_CVPR_2019_paper.pdf)]
[[Code](https://github.com/PramuPerera/OCGAN)]<br>
*Datasets: COIL100, fMNIST, MNIST, CIFAR10*<br>
*Task: Image Classification*

* RaPP: Novelty Detection with Reconstruction along Projection Pathway (ICLR 2020) 
[[Paper](https://openreview.net/forum?id=HkgeGeBYDB)]
[[Code](https://drive.google.com/drive/folders/1sknl_i4zmvSsPYZdzYxbg66ZSYDZ_abg?usp=sharing)]<br>
*Datasets: fMNIST, MNIST, MI-F and MI-V, STL, OTTO, SNSR, EOPT, NASA, RARM*<br>
*Task: Image Classification, Anomaly Detection*

* Novelty Detection Via Blurring (ICLR 2020) 
[[Paper](https://openreview.net/forum?id=ByeNra4FDB)]<br>
*Datasets: CIFAR-10, CIFAR-100, CelebA, ImageNet, LSUN, SVHN*<br>
*Task: Image Classification*

<!----------------------------------------------------------------------------------------------------------------------------------------------->
## Other Related Papers
<!----------------------------------------------------------------------------------------------------------------------------------------------->

* Understanding Cross-Domain Few-Shot Learning Based on Domain Similarity and Few-Shot Difficulty	(NeurIPS 2022) 
[[Paper](https://arxiv.org/pdf/2202.01339.pdf)]
[[Code](https://github.com/sungnyun/understanding-cdfsl)]<br>
*Datasets: ImageNet, tieredImageNet, and miniImageNet for source domain similarity to ImageNet: Places,CUB,Cars,Plantae,EuroSAT,CropDisease,ISIC,ChestX*<br>
*Task: Active Learning*

* Self-organization in a perceptual network (Info-max)(IEEE 1988) 
[[Paper](https://ieeexplore.ieee.org/document/36)]<br>


* Mutual Information-Based Temporal Difference Learning for Human Pose Estimation in Video (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.08475)]<br>

* Upcycling Models under Domain and Category Shift (CVPR 2023) 
[[Paper](https://arxiv.org/abs/2303.07110)]<br>

* Generative Meta-Adversarial Network for Unseen Object Navigation (ECCV 2022) 
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136990295.pdf)]
[[Code](https://github.com/sx-zhang/GMAN.git)]<br>
*Datasets: AI2THOR  and RoboTHOR*<br>
*Task: Object Navigation*


<!----------------------------------------------------------------------------------------------------------------------------------------------->
## Action Recognition Related
<!----------------------------------------------------------------------------------------------------------------------------------------------->
### Surveys
* Vision Transformers for Action Recognition: A Survey (Arxiv 2022) 
[[Paper](https://arxiv.org/abs/2209.05700)]


