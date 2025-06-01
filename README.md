
# Pedestrian trajectory prediction method based on graph neural networks.
<img width="1624" alt="Figure 2" src="https://github.com/user-attachments/assets/def5fc2d-af64-4101-a64e-dc99bffcc8c7" />


 # Taxonomy of Trajectory Prediction Methods based on GNNs
 <img width="2138" alt="Figure 3" src="https://github.com/user-attachments/assets/7b3ec9ac-8e66-457b-9cfc-dd8d43255847" />


 # Conventional Graph-based Methods
Conventional graph-based methods usually represent pedestrians as nodes and social relationships between pedestrians as edges, thus forming a dense graph structure. Subsequently, feature aggregation is performed through graph convolution or graph attention to model the social relationships between pedestrians, thereby improving the ability to predict future trajectories. According to the graph construction strategy, we further divide conventional graph-based methods into static, frame-wise, and spatio-temporal graph models.
<img width="1033" alt="Figure 4" src="https://github.com/user-attachments/assets/39273445-14c2-41f9-b2ec-20ef699c8a5a" />

 ### Static graph models
Kosaraju V, Sadeghian A, Martín-Martín R, et al. Social-bigat: Multimodal trajectory forecasting using bicycle-gan and graph attention networks[J]. Advances in neural information processing systems. [paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/d09bf41544a3365a46c9077ebb5e35c3-Paper.pdf)

Chen Y, Liu C, Shi B, et al. Comogcn: Coherent motion aware trajectory prediction with graph representation[J]. arXiv preprint arXiv:2005.00754, 2020. [paper](https://arxiv.org/pdf/2005.00754)

Li L, Yao J, Wenliang L, et al. Grin: Generative relation and intention network for multi-agent trajectory prediction[J]. Advances in Neural Information Processing Systems, 2021. [paper](https://proceedings.neurips.cc/paper/2021/hash/e3670ce0c315396e4836d7024abcf3dd-Abstract.html) 

Liu C, Chen Y, Liu M, et al. AVGCN: Trajectory prediction using graph convolutional networks guided by human attention[C]//2021 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9560908)

 ### Frame-wise graph models
Huang Y, Bi H, Li Z, et al. Stgat: Modeling spatial-temporal interactions for human trajectory prediction[C]//Proceedings of the IEEE/CVF international conference on computer vision. [paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_STGAT_Modeling_Spatial-Temporal_Interactions_for_Human_Trajectory_Prediction_ICCV_2019_paper.pdf) [code](https://github.com/huang-xx/STGAT)

Haddad S, Lam S K. Self-growing spatial graph networks for pedestrian trajectory prediction[C]//Proceedings of the IEEE/CVF Winter conference on applications of computer vision. 2020. [paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Haddad_Self-Growing_Spatial_Graph_Networks_for_Pedestrian_Trajectory_Prediction_WACV_2020_paper.pdf) 

Peng Y, Zhang G, Li X, et al. Stirnet: A spatial-temporal interaction-aware recursive network for human trajectory prediction[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021. [paper](https://openaccess.thecvf.com/content/ICCV2021W/SoMoF/papers/Peng_STIRNet_A_Spatial-Temporal_Interaction-Aware_Recursive_Network_for_Human_Trajectory_Prediction_ICCVW_2021_paper.pdf)

Monti A, Bertugli A, Calderara S, et al. Dag-net: Double attentive graph neural network for trajectory forecasting[C]//2020 25th international conference on pattern recognition (ICPR). IEEE, 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9412114) [code](https://github.com/alexmonti19/dagnet)

Kong W, Liu Y, Li H, et al. GSTA: Pedestrian trajectory prediction based on global spatio-temporal association of graph attention network[J]. Pattern Recognition Letters, 2022. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0167865522002057)

Zhang L, She Q, Guo P. Stochastic trajectory prediction with social graph network[J]. arXiv preprint arXiv:1907.10233, 2019. [paper](https://arxiv.org/pdf/1907.10233)

Zhou Z, Huang G, Su Z, et al. Dynamic attention-based CVAE-GAN for pedestrian trajectory prediction[J]. IEEE Robotics and Automation Letters, 2022. [paper](https://ieeexplore.ieee.org/abstract/document/9996571)

Huang L, Zhuang J, Cheng X, et al. STI-GAN: Multimodal pedestrian trajectory prediction using spatiotemporal interactions and a generative adversarial network[J]. IEEE Access, 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9387292)

Cheng H, Liu M, Chen L, et al. Gatraj: A graph-and attention-based multi-agent trajectory prediction model[J]. ISPRS Journal of Photogrammetry and Remote Sensing, 2023. [paper](https://www.sciencedirect.com/science/article/pii/S092427162300268X) [code](https://github.com/mengmengliu1998/GATraj)

  ### Spatiotemporal graph models



 # Sparse Graph-based Methods


 ### Uninterpretable sparse graph models


 ### Interpretable sparse graph models


 # Multi-Graph-based Methods


 ### Multi-feature graph models


 ### Temporal graph models


  ### Cross-spatial-temporal graph models


   ### Scene graph models
