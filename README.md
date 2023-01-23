# Open World Papers

Current machine learning algorithms are bound to a closed-world assumption. This means that these models assume the number of categories the model will encounter in the test time is predefined. However, this assumption needs to be revised in the real world. There are many challenges associated with the real-world setting that traditional machine learning models can not resolve. The uncertainty a model can encounter during test time and the strategy to address it have been considered recently under different computer vision branches. 

| **Problem Name**  |     **Problem Goal** |
|---------------|-----------|
| Zero-shot Recognition | Classify test samples from a set of unseen but known categories| 
| Generalized Zero-shot Recognition | Classify test samples from a set of seen and unseen but known categories| 
| Open-set Recognition  | Classify test samples from a set of seen categories while rejecting samples from unknown categories|
| Novel Class Discovery | Classify test samples from a set of unknown categories into proposed clasters|
| Generalized Category Discovery | Classify test samples from a set of seen or unknown categories into seen categories or proposed clasters |
| Open vocabulary | Classify test samples from a set of seen or unknown categories into proposed clasters and find the corresponding name for that claster with the help of additional information like another modality or language models.|
