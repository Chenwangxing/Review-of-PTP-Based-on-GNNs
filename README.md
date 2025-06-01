
# Pedestrian trajectory prediction method based on graph neural networks.
GNN-based pedestrian trajectory prediction methods typically adopt an encoder-decoder architecture. The encoder constructs interaction graphs based on observed trajectory data, followed by the application of graph convolution or its variants to extract spatial-temporal features of pedestrians. The decoder then generates future trajectories based on the extracted interaction features. To model different types of interactions, researchers employ various graph structures to represent social, spatiotemporal, or heterogeneous relationships among agents.
<img width="1624" alt="Figure 2" src="https://github.com/user-attachments/assets/def5fc2d-af64-4101-a64e-dc99bffcc8c7" />


 # Taxonomy of Trajectory Prediction Methods based on GNNs
 <img width="2138" alt="Figure 3" src="https://github.com/user-attachments/assets/7b3ec9ac-8e66-457b-9cfc-dd8d43255847" />


 # 1. Conventional Graph-based Methods
Conventional graph-based methods usually represent pedestrians as nodes and social relationships between pedestrians as edges, thus forming a dense graph structure. Subsequently, feature aggregation is performed through graph convolution or graph attention to model the social relationships between pedestrians, thereby improving the ability to predict future trajectories. According to the graph construction strategy, we further divide conventional graph-based methods into static, frame-wise, and spatio-temporal graph models.
<img width="1033" alt="Figure 4" src="https://github.com/user-attachments/assets/39273445-14c2-41f9-b2ec-20ef699c8a5a" />

 ### 1.1. Static graph models
Kosaraju V, Sadeghian A, Martín-Martín R, et al. Social-bigat: Multimodal trajectory forecasting using bicycle-gan and graph attention networks[J]. Advances in neural information processing systems. [paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/d09bf41544a3365a46c9077ebb5e35c3-Paper.pdf)

Chen Y, Liu C, Shi B, et al. Comogcn: Coherent motion aware trajectory prediction with graph representation[J]. arXiv preprint arXiv:2005.00754, 2020. [paper](https://arxiv.org/pdf/2005.00754)

Li L, Yao J, Wenliang L, et al. Grin: Generative relation and intention network for multi-agent trajectory prediction[J]. Advances in Neural Information Processing Systems, 2021. [paper](https://proceedings.neurips.cc/paper/2021/hash/e3670ce0c315396e4836d7024abcf3dd-Abstract.html) 

Liu C, Chen Y, Liu M, et al. AVGCN: Trajectory prediction using graph convolutional networks guided by human attention[C]//2021 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9560908)

 ### 1.2. Frame-wise graph models
Huang Y, Bi H, Li Z, et al. Stgat: Modeling spatial-temporal interactions for human trajectory prediction[C]//Proceedings of the IEEE/CVF international conference on computer vision. [paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_STGAT_Modeling_Spatial-Temporal_Interactions_for_Human_Trajectory_Prediction_ICCV_2019_paper.pdf) [code](https://github.com/huang-xx/STGAT)

Haddad S, Lam S K. Self-growing spatial graph networks for pedestrian trajectory prediction[C]//Proceedings of the IEEE/CVF Winter conference on applications of computer vision. 2020. [paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Haddad_Self-Growing_Spatial_Graph_Networks_for_Pedestrian_Trajectory_Prediction_WACV_2020_paper.pdf) 

Peng Y, Zhang G, Li X, et al. Stirnet: A spatial-temporal interaction-aware recursive network for human trajectory prediction[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021. [paper](https://openaccess.thecvf.com/content/ICCV2021W/SoMoF/papers/Peng_STIRNet_A_Spatial-Temporal_Interaction-Aware_Recursive_Network_for_Human_Trajectory_Prediction_ICCVW_2021_paper.pdf)

Monti A, Bertugli A, Calderara S, et al. Dag-net: Double attentive graph neural network for trajectory forecasting[C]//2020 25th international conference on pattern recognition (ICPR). IEEE, 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9412114) [code](https://github.com/alexmonti19/dagnet)

Kong W, Liu Y, Li H, et al. GSTA: Pedestrian trajectory prediction based on global spatio-temporal association of graph attention network[J]. Pattern Recognition Letters, 2022. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0167865522002057)

Zhang L, She Q, Guo P. Stochastic trajectory prediction with social graph network[J]. arXiv preprint arXiv:1907.10233, 2019. [paper](https://arxiv.org/pdf/1907.10233)

Zhou Z, Huang G, Su Z, et al. Dynamic attention-based CVAE-GAN for pedestrian trajectory prediction[J]. IEEE Robotics and Automation Letters, 2022. [paper](https://ieeexplore.ieee.org/abstract/document/9996571)

Huang L, Zhuang J, Cheng X, et al. STI-GAN: Multimodal pedestrian trajectory prediction using spatiotemporal interactions and a generative adversarial network[J]. IEEE Access, 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9387292)

Cheng H, Liu M, Chen L, et al. Gatraj: A graph-and attention-based multi-agent trajectory prediction model[J]. ISPRS Journal of Photogrammetry and Remote Sensing, 2023. [paper](https://www.sciencedirect.com/science/article/pii/S092427162300268X) [code](https://github.com/mengmengliu1998/GATraj)

  ### 1.3. Spatiotemporal graph models
Sun Y, He T, Hu J, et al. Socially-aware graph convolutional network for human trajectory prediction[C]//2019 IEEE 3rd Information Technology, Networking, Electronic and Automation Control Conference (ITNEC). IEEE. [paper](https://ieeexplore.ieee.org/abstract/document/8729387)

Mohamed A, Qian K, Elhoseiny M, et al. Social-stgcnn: A social spatio-temporal graph convolutional neural network for human trajectory prediction[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020. [paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Mohamed_Social-STGCNN_A_Social_Spatio-Temporal_Graph_Convolutional_Neural_Network_for_Human_CVPR_2020_paper.pdf) [code](https://github.com/abduallahmohamed/Social-STGCNN)

Xue H, Huynh D Q, Reynolds M. Take a nap: Non-autoregressive prediction for pedestrian trajectories[C]//International Conference on Neural Information Processing. Cham: Springer International Publishing, 2020. [paper](https://link.springer.com/chapter/10.1007/978-3-030-63830-6_46) 

Wu Y, Chen G, Li Z, et al. HSTA: A hierarchical spatio-temporal attention model for trajectory prediction[J]. IEEE Transactions on Vehicular Technology, 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9548801)

Li K, Eiffert S, Shan M, et al. Attentional-GCNN: Adaptive pedestrian trajectory prediction towards generic autonomous vehicle use cases[C]//2021 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2021: 14241-14247. [paper](https://ieeexplore.ieee.org/abstract/document/9561480)

Wang C, Cai S, Tan G. Graphtcn: Spatio-temporal interaction modeling for human trajectory prediction[C]//Proceedings of the IEEE/CVF winter conference on applications of computer vision. 2021. [paper](https://openaccess.thecvf.com/content/WACV2021/papers/Wang_GraphTCN_Spatio-Temporal_Interaction_Modeling_for_Human_Trajectory_Prediction_WACV_2021_paper.pdf)

Zhao Z, Liu C. STUGCN: A social spatio-temporal unifying graph convolutional network for trajectory prediction[C]//2021 6th International Conference on Automation, Control and Robotics Engineering (CACRE). IEEE, 2021: 546-550. [paper](https://ieeexplore.ieee.org/abstract/document/9501325)

Lian J, Yu F, Li L, et al. Causal temporal–spatial pedestrian trajectory prediction with goal point estimation and contextual interaction[J]. IEEE Transactions on Intelligent Transportation Systems, 2022. [paper](https://ieeexplore.ieee.org/abstract/document/9896809)

Luo T, Shang H, Li Z, et al. Multi-Dimensional Spatial-Temporal Fusion for Pedestrian Trajectory Prediction[C]//2022 2nd International Conference on Networking Systems of AI (INSAI). IEEE, 2022: 170-174. [paper](https://ieeexplore.ieee.org/abstract/document/10175121)

Tang H, Wei P, Li J, et al. Evostgat: Evolving spatiotemporal graph attention networks for pedestrian trajectory prediction[J]. Neurocomputing, 2022. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231222003460)

Li Y, Cui J, Zhao Z, et al. ST-AGNN: Spatial-Temporal Attention Graph Neural Network for Pedestrian Trajectory Prediction[M]//Applied Mathematics, Modeling and Computer Simulation. IOS Press, 2022: 268-275. [paper](https://ebooks.iospress.nl/doi/10.3233/ATDE221042)

Yang J, Sun X, Wang R G, et al. PTPGC: Pedestrian trajectory prediction by graph attention network with ConvLSTM[J]. Robotics and Autonomous Systems, 2022. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0921889021002165)

Zhu W, Liu Y, Wang P, et al. Tri-HGNN: Learning triple policies fused hierarchical graph neural networks for pedestrian trajectory prediction[J]. Pattern Recognition, 2023. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320323004703)

Wang R, Hu Z, Song X, et al. Trajectory distribution aware graph convolutional network for trajectory prediction considering spatio-temporal interactions and scene information[J]. IEEE Transactions on Knowledge and Data Engineering, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10309163)

Lv P, Wang W, Wang Y, et al. SSAGCN: social soft attention graph convolution network for pedestrian trajectory prediction[J]. IEEE transactions on neural networks and learning systems, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10063206) [code](https://github.com/WW-Tong/ssagcn_for_path_prediction)

Yang C, Pei Z. Long-short term spatio-temporal aggregation for trajectory prediction[J]. IEEE Transactions on Intelligent Transportation Systems, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10018105)

Zhi C Y, Sun H J, Xu T. Adaptive trajectory prediction without catastrophic forgetting[J]. The Journal of Supercomputing, 2023. [paper](https://link.springer.com/article/10.1007/s11227-023-05241-z) 

Lian J, Ren W, Li L, et al. Ptp-stgcn: pedestrian trajectory prediction based on a spatio-temporal graph convolutional neural network[J]. Applied Intelligence, 2023. [paper](https://link.springer.com/article/10.1007/s10489-022-03524-1)

Bhujel N, Yau W Y. Disentangling crowd interactions for pedestrians trajectory prediction[J]. IEEE Robotics and Automation Letters, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10083225)

Chen X, Luo F, Zhao F, et al. Goal-Guided and Interaction-Aware State Refinement Graph Attention Network for Multi-Agent Trajectory Prediction[J]. IEEE Robotics and Automation Letters, 2023, 9(1): 57-64. [paper](https://ieeexplore.ieee.org/abstract/document/10313963)

Liu Y, Li B, Wang X, et al. Attention-aware Social Graph Transformer Networks for Stochastic Trajectory Prediction[J]. IEEE Transactions on Knowledge and Data Engineering, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10504962)

Yang B, Fan F, Ni R, et al. A multi-task learning network with a collision-aware graph transformer for traffic-agents trajectory prediction[J]. IEEE Transactions on Intelligent Transportation Systems, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10379537) 

Khel M H K, Greaney P, McAfee M, et al. GSTGM: Graph, spatial–temporal attention and generative based model for pedestrian multi-path prediction[J]. Image and Vision Computing, 2024: 105245. [paper](https://www.sciencedirect.com/science/article/pii/S0262885624003500)

Yang X, Fan J, Wang X, et al. CSGAT-Net: a conditional pedestrian trajectory prediction network based on scene semantic maps and spatiotemporal graph attention[J]. Neural Computing and Applications, 2024: 1-15. [paper](https://link.springer.com/article/10.1007/s00521-024-09784-x)

Ruan K, Di X. InfoSTGCAN: An Information-Maximizing Spatial-Temporal Graph Convolutional Attention Network for Heterogeneous Human Trajectory Prediction[J]. Computers, 2024, 13(6): 151. [paper](https://www.mdpi.com/2073-431X/13/6/151)

Mi J, Zhang X, Zeng H, et al. DERGCN: Dynamic-Evolving graph convolutional networks for human trajectory prediction[J]. Neurocomputing, 2024. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231223012407)

 # 2. Sparse Graph-based Methods
Sparse graph-based methods dynamically select the most relevant neighbors for each pedestrian to construct a sparse graph structure. By reducing redundant connections, these methods mitigate interference from irrelevant interactions and enhance model efficiency and prediction performance. We divide sparse graph-based methods into uninterpretable and interpretable sparse graph models based on whether the interaction filtering mechanism incorporates interpretable priors.
<img width="733" alt="Figure 6" src="https://github.com/user-attachments/assets/b1b39e1e-b7db-470c-868a-35a1981bbddf" />


 ### 2.1. Uninterpretable sparse graph models
Shi L, Wang L, Long C, et al. SGCN: Sparse graph convolution network for pedestrian trajectory prediction[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021. [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Shi_SGCN_Sparse_Graph_Convolution_Network_for_Pedestrian_Trajectory_Prediction_CVPR_2021_paper.pdf) [code](https://github.com/shuaishiliu/SGCN)

Zhou Y, Wu H, Cheng H, et al. Social graph convolutional LSTM for pedestrian trajectory prediction[J]. IET Intelligent Transport Systems, 2021. [paper](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/itr2.12033)

Wu Y, Wang L, Zhou S, et al. Multi-stream representation learning for pedestrian trajectory prediction[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2023. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/25389) [code](https://github.com/YuxuanIAIR/MSRL-master)


 ### 2.2. Interpretable sparse graph models
Sun J, Jiang Q, Lu C. Recursive social behavior graph for trajectory prediction[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020. [paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sun_Recursive_Social_Behavior_Graph_for_Trajectory_Prediction_CVPR_2020_paper.pdf) 

Pedestrian Trajectory Prediction Based on Improved Social Spatio-Temporal Graph Convolution Neural Network[C]//Proceedings of the 2022 5th International Conference on Machine Learning and Natural Language Processing. 2022: 63-67. [paper](https://dl.acm.org/doi/abs/10.1145/3578741.3578754)

Zhang X, Angeloudis P, Demiris Y. Dual-branch spatio-temporal graph neural networks for pedestrian trajectory prediction[J]. Pattern Recognition, 2023. [paper](https://www.sciencedirect.com/science/article/pii/S0031320323003345)

Lv K, Yuan L. SKGACN: social knowledge-guided graph attention convolutional network for human trajectory prediction[J]. IEEE Transactions on Instrumentation and Measurement, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10145416)

Li X, Zhang Q, Wang W, et al. SA-GCNN: Spatial Attention Based Graph Convolutional Neural Network for Pedestrian Trajectory Prediction[C]//2023 IEEE International Conference on Robotics and Biomimetics (ROBIO). IEEE, 2023: 1-6. [paper](https://ieeexplore.ieee.org/abstract/document/10354667)

Sun C, Wang B, Leng J, et al. SDAGCN: Sparse Directed Attention Graph Convolutional Network for Spatial Interaction in Pedestrian Trajectory Prediction[J]. IEEE Internet of Things Journal, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10547191)

Chen W, Sang H, Wang J, et al. IMGCN: interpretable masked graph convolution network for pedestrian trajectory prediction[J]. Transportmetrica B: Transport Dynamics, 2024. [paper](https://www.tandfonline.com/doi/abs/10.1080/21680566.2024.2389896) [code](https://github.com/Chenwangxing/IMGCN_master)


 # Multi-Graph-based Methods


 ### Multi-feature graph models


 ### Temporal graph models


  ### Cross-spatial-temporal graph models


   ### Scene graph models
