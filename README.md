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
- [Zero-Shot Learning](https://github.com/SarahRastegar/Open-World-Papers/blob/main/Zero-Shot.md)  
- [Out-of-Distribution Detection](https://github.com/SarahRastegar/Open-World-Papers/blob/main/OOD.md) 
- [Open-Set Recognition](https://github.com/SarahRastegar/Open-World-Papers/blob/main/OSR.md)
- [Novel Class Discovery](https://github.com/SarahRastegar/Open-World-Papers/blob/main/NCD.md))
- [Open Vocabulary](https://github.com/SarahRastegar/Open-World-Papers/blob/main/Open-Vocabulary.md)
- [Fine Grained](https://github.com/SarahRastegar/Open-World-Papers/blob/main/Fine-Grained.md)
- [Long Tail](https://github.com/SarahRastegar/Open-World-Papers/blob/main/Long-Tailed.md)
- [Anomaly Detection](https://github.com/SarahRastegar/Open-World-Papers/blob/main/Anomaly-Detection.md)
- [Novelty Detection](https://github.com/SarahRastegar/Open-World-Papers/blob/main/ND.md)
- [Video Open World Papers](https://github.com/SarahRastegar/Open-World-Papers/blob/main/Video-Open-World.md)


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

In this repository section [Zero-Shot Learning](https://github.com/SarahRastegar/Open-World-Papers/blob/main/Zero-Shot.md) and [Zero-Shot Learning Videos](https://github.com/SarahRastegar/Open-World-Papers/blob/main/Video-Open-World.md#Zero-Shot-Learning-Videos) has summarized the recent papers for both *Zero-shot* and *Generalized Zero-shot*.

### Unknown Categories
Telsa scientists are relieved that their autopilot model is ready to be employed in the real world. Their model can handle unseen environments and unseen actions and categories. However, there is still a minor problem; there is one extremely likely prediction that one day their autopilot will encounter objects or actions, they are unaware of at training time. So they decide to train their model to make them aware anytime it sees something new or rejects it. First, however, they need to learn how to regulate this uncertainty. Some scientists suggest that the model rejects anything from the training time distribution. In computer vision, this problem is known as [Out-of-Distribution Detection](https://github.com/SarahRastegar/Open-World-Papers/blob/main/OOD.md) or its video version as [Out-of-Distribution Detection Videos](https://github.com/SarahRastegar/Open-World-Papers/blob/main/Video-Open-World.md#Out-of-Distribution-Detection-Videos). Another highly overlapped scenario is [Anomaly Detection](https://github.com/SarahRastegar/Open-World-Papers/blob/main/Anomaly-Detection.md) or its video version [Anomaly Detection Videos](https://github.com/SarahRastegar/Open-World-Papers/blob/main/Video-Open-World.md#Anomaly-Detection-Videos). In Anomaly detection, the anomal object or events usually is rare, and we are interested in extracting them from data for OOD. However, the rejected samples are typically abundant, and the rejection can be done automatically. 

Sometimes it is essential to reject data that we need to know their category while they could be in distribution. This problem is called [Open-Set Recognition](https://github.com/SarahRastegar/Open-World-Papers/blob/main/OSR.md), and its video [Open-Set Recognition Videos](https://github.com/SarahRastegar/Open-World-Papers/blob/main/Video-Open-World.md#Open-Set-Recognition-Videos). 

The problem with OOD or OSR is that the model rejects the unknowns, so though it could improve itself, it does not learn the new categories, which is in direct contrast with humans, which start which categories of new objects or actions. So if an agent can detect these novel categories and cluster them, it could use this novel knowledge to its benefit and use from the vast number of novel but unlabelled samples. This problem is called [Novel Class Discovery](https://github.com/SarahRastegar/Open-World-Papers/blob/main/NCD.md) or its video version [Novel Class Discovery Videos](https://github.com/SarahRastegar/Open-World-Papers/blob/main/Video-Open-World.md#Novel-Class-Discovery-Videos). Unfortunately, when there is a mixture of unknown and known categories in test time, the model is prone to assign samples mostly to either known or unknown categories. **Generalized Category Discovery** addresses this problem when encountering novel classes alongside already seen classes in the test time. 

While novel class discovery and its cousin generalized category discovery have a lot of challenges, dealing with fine-grained categories where the categories have subtle differences and may have a lot of overlap or long-tail distribution where the frequency of each category can differ drastically elevates these obstacles to a very dark realm. So in this repository, we also consider the works that are done in [Fine Grained](https://github.com/SarahRastegar/Open-World-Papers/blob/main/Fine-Grained.md) or [Fine Grained Videos](https://github.com/SarahRastegar/Open-World-Papers/blob/main/Video-Open-World.md#Fine-Grained-Videos). While Long-tail is still in its infancy, especially for the video field, the challenge it introduces to the open world is very real, therefore works that focus on [Long Tail](https://github.com/SarahRastegar/Open-World-Papers/blob/main/Long-Tailed.md) or [Long Tail Videos](https://github.com/SarahRastegar/Open-World-Papers/blob/main/Video-Open-World.md#Long-Tail-Videos) also has been collected. 

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


