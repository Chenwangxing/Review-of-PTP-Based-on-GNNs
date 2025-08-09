
# Pedestrian trajectory prediction method based on graph neural networks.

## The architecture of the pedestrian trajectory prediction method based on GNNs.
GNN-based pedestrian trajectory prediction methods typically adopt an encoder-decoder architecture. The encoder constructs interaction graphs based on observed trajectory data, followed by the application of graph convolution or its variants to extract spatial-temporal features of pedestrians. The decoder then generates future trajectories based on the extracted interaction features. To model different types of interactions, researchers employ various graph structures to represent social, spatiotemporal, or heterogeneous relationships among agents.

<img width="2000" alt="Figure 2" src="https://github.com/user-attachments/assets/def5fc2d-af64-4101-a64e-dc99bffcc8c7">


 ## Taxonomy of Trajectory Prediction Methods based on GNNs
 According to the differences in graph construction strategies and interaction modeling paradigms, we categorize existing methods into five types：**1. Conventional graph-based methods**；**2. Sparse graph-based methods**；**3. Multi-graph-based methods**；**4. Heterogeneous graph-based methods**；**5. High-order graph-based methods**.
 
<img width="1500" alt="Figure 3" src="https://github.com/user-attachments/assets/60a1aa74-b318-4082-9caa-5fa6380939eb" />

 
In addition, we present the timeline of various GNN-based trajectory prediction methods along with some representative methods.

<img width="2000" alt="1发展时间图" src="https://github.com/user-attachments/assets/d986cecb-831a-4c32-b01d-2244318efab9" />


 ## 1. Conventional Graph-based Methods
Conventional graph-based methods usually represent pedestrians as nodes and social relationships between pedestrians as edges, thus forming a dense graph structure. Subsequently, feature aggregation is then performed through the graph convolutional network (GCN) or graph attention network (GAT) to capture the social relationships among pedestrians, thereby improving the accuracy of trajectory prediction. According to the graph construction strategy, we further divide conventional graph-based methods into **static graph models**, **frame-wise graph models**, and **spatio-temporal graph models**.

 <img width="1000" alt="Figure 4" src="https://github.com/user-attachments/assets/53536bbb-15d2-4b36-ada4-d8dcb9a95408" />


 ### 1.1. Static graph models
- Kosaraju V, Sadeghian A, Martín-Martín R, et al. Social-bigat: Multimodal trajectory forecasting using bicycle-gan and graph attention networks[J]. Advances in neural information processing systems. [paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/d09bf41544a3365a46c9077ebb5e35c3-Paper.pdf)

- Chen Y, Liu C, Shi B, et al. Comogcn: Coherent motion aware trajectory prediction with graph representation[J]. arXiv preprint arXiv:2005.00754, 2020. [paper](https://arxiv.org/pdf/2005.00754)

- Li L, Yao J, Wenliang L, et al. Grin: Generative relation and intention network for multi-agent trajectory prediction[J]. Advances in Neural Information Processing Systems, 2021. [paper](https://proceedings.neurips.cc/paper/2021/hash/e3670ce0c315396e4836d7024abcf3dd-Abstract.html) 

- Liu C, Chen Y, Liu M, et al. AVGCN: Trajectory prediction using graph convolutional networks guided by human attention[C]//2021 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9560908)

 ### 1.2. Frame-wise graph models
- Huang Y, Bi H, Li Z, et al. Stgat: Modeling spatial-temporal interactions for human trajectory prediction[C]//Proceedings of the IEEE/CVF international conference on computer vision. [paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_STGAT_Modeling_Spatial-Temporal_Interactions_for_Human_Trajectory_Prediction_ICCV_2019_paper.pdf) [code](https://github.com/huang-xx/STGAT)

- Haddad S, Lam S K. Self-growing spatial graph networks for pedestrian trajectory prediction[C]//Proceedings of the IEEE/CVF Winter conference on applications of computer vision. 2020. [paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Haddad_Self-Growing_Spatial_Graph_Networks_for_Pedestrian_Trajectory_Prediction_WACV_2020_paper.pdf) 

- Peng Y, Zhang G, Li X, et al. Stirnet: A spatial-temporal interaction-aware recursive network for human trajectory prediction[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021. [paper](https://openaccess.thecvf.com/content/ICCV2021W/SoMoF/papers/Peng_STIRNet_A_Spatial-Temporal_Interaction-Aware_Recursive_Network_for_Human_Trajectory_Prediction_ICCVW_2021_paper.pdf)

- Monti A, Bertugli A, Calderara S, et al. Dag-net: Double attentive graph neural network for trajectory forecasting[C]//2020 25th international conference on pattern recognition (ICPR). IEEE, 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9412114) [code](https://github.com/alexmonti19/dagnet)

- Kong W, Liu Y, Li H, et al. GSTA: Pedestrian trajectory prediction based on global spatio-temporal association of graph attention network[J]. Pattern Recognition Letters, 2022. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0167865522002057)

- Zhang L, She Q, Guo P. Stochastic trajectory prediction with social graph network[J]. arXiv preprint arXiv:1907.10233, 2019. [paper](https://arxiv.org/pdf/1907.10233)

- Zhou Z, Huang G, Su Z, et al. Dynamic attention-based CVAE-GAN for pedestrian trajectory prediction[J]. IEEE Robotics and Automation Letters, 2022. [paper](https://ieeexplore.ieee.org/abstract/document/9996571)

- Huang L, Zhuang J, Cheng X, et al. STI-GAN: Multimodal pedestrian trajectory prediction using spatiotemporal interactions and a generative adversarial network[J]. IEEE Access, 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9387292)

- Cheng H, Liu M, Chen L, et al. Gatraj: A graph-and attention-based multi-agent trajectory prediction model[J]. ISPRS Journal of Photogrammetry and Remote Sensing, 2023. [paper](https://www.sciencedirect.com/science/article/pii/S092427162300268X) [code](https://github.com/mengmengliu1998/GATraj)

  ### 1.3. Spatiotemporal graph models
- Sun Y, He T, Hu J, et al. Socially-aware graph convolutional network for human trajectory prediction[C]//2019 IEEE 3rd Information Technology, Networking, Electronic and Automation Control Conference (ITNEC). IEEE. [paper](https://ieeexplore.ieee.org/abstract/document/8729387)

- Mohamed A, Qian K, Elhoseiny M, et al. Social-stgcnn: A social spatio-temporal graph convolutional neural network for human trajectory prediction[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020. [paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Mohamed_Social-STGCNN_A_Social_Spatio-Temporal_Graph_Convolutional_Neural_Network_for_Human_CVPR_2020_paper.pdf) [code](https://github.com/abduallahmohamed/Social-STGCNN)

- Xue H, Huynh D Q, Reynolds M. Take a nap: Non-autoregressive prediction for pedestrian trajectories[C]//International Conference on Neural Information Processing. Cham: Springer International Publishing, 2020. [paper](https://link.springer.com/chapter/10.1007/978-3-030-63830-6_46) 

- Wu Y, Chen G, Li Z, et al. HSTA: A hierarchical spatio-temporal attention model for trajectory prediction[J]. IEEE Transactions on Vehicular Technology, 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9548801)

- Li K, Eiffert S, Shan M, et al. Attentional-GCNN: Adaptive pedestrian trajectory prediction towards generic autonomous vehicle use cases[C]//2021 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2021: 14241-14247. [paper](https://ieeexplore.ieee.org/abstract/document/9561480)

- Wang C, Cai S, Tan G. Graphtcn: Spatio-temporal interaction modeling for human trajectory prediction[C]//Proceedings of the IEEE/CVF winter conference on applications of computer vision. 2021. [paper](https://openaccess.thecvf.com/content/WACV2021/papers/Wang_GraphTCN_Spatio-Temporal_Interaction_Modeling_for_Human_Trajectory_Prediction_WACV_2021_paper.pdf)

- Zhao Z, Liu C. STUGCN: A social spatio-temporal unifying graph convolutional network for trajectory prediction[C]//2021 6th International Conference on Automation, Control and Robotics Engineering (CACRE). IEEE, 2021: 546-550. [paper](https://ieeexplore.ieee.org/abstract/document/9501325)

- Lian J, Yu F, Li L, et al. Causal temporal–spatial pedestrian trajectory prediction with goal point estimation and contextual interaction[J]. IEEE Transactions on Intelligent Transportation Systems, 2022. [paper](https://ieeexplore.ieee.org/abstract/document/9896809)

- Luo T, Shang H, Li Z, et al. Multi-Dimensional Spatial-Temporal Fusion for Pedestrian Trajectory Prediction[C]//2022 2nd International Conference on Networking Systems of AI (INSAI). IEEE, 2022: 170-174. [paper](https://ieeexplore.ieee.org/abstract/document/10175121)

- Tang H, Wei P, Li J, et al. Evostgat: Evolving spatiotemporal graph attention networks for pedestrian trajectory prediction[J]. Neurocomputing, 2022. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231222003460)

- Li Y, Cui J, Zhao Z, et al. ST-AGNN: Spatial-Temporal Attention Graph Neural Network for Pedestrian Trajectory Prediction[M]//Applied Mathematics, Modeling and Computer Simulation. IOS Press, 2022: 268-275. [paper](https://ebooks.iospress.nl/doi/10.3233/ATDE221042)

- Yang J, Sun X, Wang R G, et al. PTPGC: Pedestrian trajectory prediction by graph attention network with ConvLSTM[J]. Robotics and Autonomous Systems, 2022. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0921889021002165)

- Zhu W, Liu Y, Wang P, et al. Tri-HGNN: Learning triple policies fused hierarchical graph neural networks for pedestrian trajectory prediction[J]. Pattern Recognition, 2023. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320323004703)

- Wang R, Hu Z, Song X, et al. Trajectory distribution aware graph convolutional network for trajectory prediction considering spatio-temporal interactions and scene information[J]. IEEE Transactions on Knowledge and Data Engineering, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10309163)

- Lv P, Wang W, Wang Y, et al. SSAGCN: social soft attention graph convolution network for pedestrian trajectory prediction[J]. IEEE transactions on neural networks and learning systems, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10063206) [code](https://github.com/WW-Tong/ssagcn_for_path_prediction)

- Yang C, Pei Z. Long-short term spatio-temporal aggregation for trajectory prediction[J]. IEEE Transactions on Intelligent Transportation Systems, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10018105)

- Zhi C Y, Sun H J, Xu T. Adaptive trajectory prediction without catastrophic forgetting[J]. The Journal of Supercomputing, 2023. [paper](https://link.springer.com/article/10.1007/s11227-023-05241-z) 

- Lian J, Ren W, Li L, et al. Ptp-stgcn: pedestrian trajectory prediction based on a spatio-temporal graph convolutional neural network[J]. Applied Intelligence, 2023. [paper](https://link.springer.com/article/10.1007/s10489-022-03524-1)

- Bhujel N, Yau W Y. Disentangling crowd interactions for pedestrians trajectory prediction[J]. IEEE Robotics and Automation Letters, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10083225)

- Chen X, Luo F, Zhao F, et al. Goal-Guided and Interaction-Aware State Refinement Graph Attention Network for Multi-Agent Trajectory Prediction[J]. IEEE Robotics and Automation Letters, 2023, 9(1): 57-64. [paper](https://ieeexplore.ieee.org/abstract/document/10313963)

- Liu Y, Li B, Wang X, et al. Attention-aware Social Graph Transformer Networks for Stochastic Trajectory Prediction[J]. IEEE Transactions on Knowledge and Data Engineering, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10504962)

- Yang B, Fan F, Ni R, et al. A multi-task learning network with a collision-aware graph transformer for traffic-agents trajectory prediction[J]. IEEE Transactions on Intelligent Transportation Systems, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10379537) 

- Khel M H K, Greaney P, McAfee M, et al. GSTGM: Graph, spatial–temporal attention and generative based model for pedestrian multi-path prediction[J]. Image and Vision Computing, 2024: 105245. [paper](https://www.sciencedirect.com/science/article/pii/S0262885624003500)

- Yang X, Fan J, Wang X, et al. CSGAT-Net: a conditional pedestrian trajectory prediction network based on scene semantic maps and spatiotemporal graph attention[J]. Neural Computing and Applications, 2024: 1-15. [paper](https://link.springer.com/article/10.1007/s00521-024-09784-x)

- Ruan K, Di X. InfoSTGCAN: An Information-Maximizing Spatial-Temporal Graph Convolutional Attention Network for Heterogeneous Human Trajectory Prediction[J]. Computers, 2024, 13(6): 151. [paper](https://www.mdpi.com/2073-431X/13/6/151)

- Mi J, Zhang X, Zeng H, et al. DERGCN: Dynamic-Evolving graph convolutional networks for human trajectory prediction[J]. Neurocomputing, 2024. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231223012407)

 ## 2. Sparse Graph-based Methods
Sparse graph-based methods dynamically select the most relevant neighbors for each pedestrian to construct a sparse graph structure. By reducing redundant connections, these methods mitigate interference from irrelevant interactions and enhance model efficiency and prediction performance. We divide sparse graph-based methods into **uninterpretable sparse graph models** and **interpretable sparse graph models** based on whether the interaction filtering mechanism incorporates interpretable priors.

<img width="1000" alt="Figure 6" src="https://github.com/user-attachments/assets/b1b39e1e-b7db-470c-868a-35a1981bbddf" />


 ### 2.1. Uninterpretable sparse graph models
- Shi L, Wang L, Long C, et al. SGCN: Sparse graph convolution network for pedestrian trajectory prediction[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021. [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Shi_SGCN_Sparse_Graph_Convolution_Network_for_Pedestrian_Trajectory_Prediction_CVPR_2021_paper.pdf) [code](https://github.com/shuaishiliu/SGCN)

- Zhou Y, Wu H, Cheng H, et al. Social graph convolutional LSTM for pedestrian trajectory prediction[J]. IET Intelligent Transport Systems, 2021. [paper](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/itr2.12033)

- Wu Y, Wang L, Zhou S, et al. Multi-stream representation learning for pedestrian trajectory prediction[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2023. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/25389) [code](https://github.com/YuxuanIAIR/MSRL-master)


 ### 2.2. Interpretable sparse graph models
- Sun J, Jiang Q, Lu C. Recursive social behavior graph for trajectory prediction[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020. [paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sun_Recursive_Social_Behavior_Graph_for_Trajectory_Prediction_CVPR_2020_paper.pdf) 

- Pedestrian Trajectory Prediction Based on Improved Social Spatio-Temporal Graph Convolution Neural Network[C]//Proceedings of the 2022 5th International Conference on Machine Learning and Natural Language Processing. 2022: 63-67. [paper](https://dl.acm.org/doi/abs/10.1145/3578741.3578754)

- Zhang X, Angeloudis P, Demiris Y. Dual-branch spatio-temporal graph neural networks for pedestrian trajectory prediction[J]. Pattern Recognition, 2023. [paper](https://www.sciencedirect.com/science/article/pii/S0031320323003345)

- Lv K, Yuan L. SKGACN: social knowledge-guided graph attention convolutional network for human trajectory prediction[J]. IEEE Transactions on Instrumentation and Measurement, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10145416)

- Li X, Zhang Q, Wang W, et al. SA-GCNN: Spatial Attention Based Graph Convolutional Neural Network for Pedestrian Trajectory Prediction[C]//2023 IEEE International Conference on Robotics and Biomimetics (ROBIO). IEEE, 2023: 1-6. [paper](https://ieeexplore.ieee.org/abstract/document/10354667)

- Sun C, Wang B, Leng J, et al. SDAGCN: Sparse Directed Attention Graph Convolutional Network for Spatial Interaction in Pedestrian Trajectory Prediction[J]. IEEE Internet of Things Journal, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10547191)

- Chen W, Sang H, Wang J, et al. IMGCN: interpretable masked graph convolution network for pedestrian trajectory prediction[J]. Transportmetrica B: Transport Dynamics, 2024. [paper](https://www.tandfonline.com/doi/abs/10.1080/21680566.2024.2389896) [code](https://github.com/Chenwangxing/IMGCN_master)


 ## 3. Multi-Graph-based Methods
Multi-graph-based methods construct multiple graph structures to simultaneously model different interaction relationships, such as social relationships, temporal relationships, or scene semantic constraints, and achieve joint modeling through a fusion mechanism. We classify these methods based on the modeling emphasis into **multi-feature graph models**, **temporal graph models**, **cross-spatial-temporal graph models**, and **scene graph models**.

<img width="1000" alt="Figure 7" src="https://github.com/user-attachments/assets/797f2d36-8ac9-4173-b6aa-afe68b55d975" />


 ### 3.1. Multi-feature graph models
- Biswas A, Morris B T. TAGCN: topology-aware graph convolutional network for trajectory prediction[C]//Advances in Visual Computing: 15th International Symposium, ISVC 2020, San Diego, CA, USA, October 5–7, 2020, Proceedings, Part I 15. Springer International Publishing, 2020: 542-553.[paper](https://link.springer.com/chapter/10.1007/978-3-030-64556-4_42) 

- Peng Y, Zhang G, Li X, et al. SRGAT: Social relational graph attention network for human trajectory prediction[C]//Neural Information Processing: 28th International Conference, ICONIP 2021, Sanur, Bali, Indonesia, December 8–12, 2021, Proceedings, Part II 28. Springer International Publishing, 2021. [paper](https://link.springer.com/chapter/10.1007/978-3-030-92270-2_54)

- Su Y, Du J, Li Y, et al. Trajectory forecasting based on prior-aware directed graph convolutional neural network[J]. IEEE Transactions on Intelligent Transportation Systems, 2022. [paper](https://ieeexplore.ieee.org/abstract/document/9686621)

- Li M, Chen T, Du H. Trajectory prediction of cyclist based on spatial‐temporal multi‐graph network in crowded scenarios[J]. Electronics Letters, 2022, 58(3): 97-99. [paper](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ell2.12374)

- Zhou L, Zhao Y, Yang D, et al. Gchgat: Pedestrian trajectory prediction using group constrained hierarchical graph attention networks[J]. Applied Intelligence, 2022. [paper](https://link.springer.com/article/10.1007/s10489-021-02997-w)

- Bae I, Park J H, Jeon H G. Learning pedestrian group representations for multi-modal trajectory prediction[C]//European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022: 270-289. [paper](https://link.springer.com/chapter/10.1007/978-3-031-20047-2_16) [code](https://github.com/inhwanbae/GPGraph)

- Zhang X, Angeloudis P, Demiris Y. Dual-branch spatio-temporal graph neural networks for pedestrian trajectory prediction[J]. Pattern Recognition, 2023. [paper](https://www.sciencedirect.com/science/article/pii/S0031320323003345)

- Zhou H, Yang X, Fan M, et al. Static-dynamic global graph representation for pedestrian trajectory prediction[J]. Knowledge-Based Systems, 2023. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705123005257)

- Pi L, Zhang Q, Yang L, et al. Social interaction model enhanced with speculation stage for human trajectory prediction[J]. Robotics and Autonomous Systems, 2023, 161: 104352. [paper](https://www.sciencedirect.com/science/article/abs/pii/S092188902200241X)

- Yang Z, Pang C, Zeng X. Trajectory Forecasting Using Graph Convolutional Neural Networks Based on Prior Awareness and Information Fusion[J]. ISPRS International Journal of Geo-Information, 2023. [paper](https://www.mdpi.com/2220-9964/12/2/77)

- Du Q, Wang X, Yin S, et al. Social force embedded mixed graph convolutional network for multi-class trajectory prediction[J]. IEEE Transactions on Intelligent Vehicles, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10415371)

- Sheng Z, Huang Z, Chen S. Ego‐planning‐guided multi‐graph convolutional network for heterogeneous agent trajectory prediction[J]. Computer‐Aided Civil and Infrastructure Engineering, 2024. [paper](https://onlinelibrary.wiley.com/doi/full/10.1111/mice.13301)

- Lin X, Zhang Y, Wang S, et al. Multi-scale wavelet transform enhanced graph neural network for pedestrian trajectory prediction[J]. Physica A: Statistical Mechanics and its Applications, 2025. [paper](https://www.sciencedirect.com/science/article/abs/pii/S037843712400829X)

- Su Z, Huang G, Zhou Z, et al. Improving generative trajectory prediction via collision-free modeling and goal scene reconstruction[J]. Pattern Recognition Letters, 2025, 188: 117-124. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0167865524003593)

- hou D, Gao Y, Li H, et al. Group commonality graph: Multimodal pedestrian trajectory prediction via deep group features[J]. Pattern Recognition Letters, 2025. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0167865525001114)


 ### 3.2. Temporal graph models
- Yu C, Ma X, Ren J, et al. Spatio-temporal graph transformer networks for pedestrian trajectory prediction[C]//Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XII 16. Springer International Publishing, 2020. [paper](https://link.springer.com/chapter/10.1007/978-3-030-58610-2_30) [code](https://github.com/cunjunyu/STAR)
 
- Zhou H, Ren D, Xia H, et al. Ast-gnn: An attention-based spatio-temporal graph neural network for interaction-aware pedestrian trajectory prediction[J]. Neurocomputing, 2021. [paper](https://www.sciencedirect.com/science/article/abs/pii/S092523122100388X)

- Liu Z, He L, Yuan L, et al. STAGP: Spatio-Temporal Adaptive Graph Pooling Network for Pedestrian Trajectory Prediction[J]. IEEE Robotics and Automation Letters, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10373049)

- Liu Y, Zhang Y, Li K, et al. Knowledge-aware Graph Transformer for Pedestrian Trajectory Prediction[C]//2023 IEEE 26th International Conference on Intelligent Transportation Systems (ITSC). IEEE, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10421989)

- Sang H, Chen W, Wang J, et al. RDGCN: Reasonably dense graph convolution network for pedestrian trajectory prediction[J]. Measurement, 2023. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0263224123002397) 

- Zhong X, Yan X, Yang Z, et al. Visual exposes you: pedestrian trajectory prediction meets visual intention[J]. IEEE Transactions on Intelligent Transportation Systems, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10103218)

- Li J, Yang L, Chen Y, et al. MFAN: Mixing Feature Attention Network for trajectory prediction[J]. Pattern Recognition, 2024. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320323006957) [code](https://github.com/ME-SJTU/MFAN#mfan)

- Chen W, Sang H, Wang J, et al. WTGCN: wavelet transform graph convolution network for pedestrian trajectory prediction[J]. International Journal of Machine Learning and Cybernetics, 2024. [paper](https://link.springer.com/article/10.1007/s13042-024-02258-5)

- Chen W, Sang H, Wang J, et al. IMGCN: interpretable masked graph convolution network for pedestrian trajectory prediction[J]. Transportmetrica B: Transport Dynamics, 2024. [paper](https://www.tandfonline.com/doi/abs/10.1080/21680566.2024.2389896) [code](https://github.com/Chenwangxing/IMGCN_master)

- Gao L, Gu X, Chen F, et al. Sparse Transformer Network with Spatial-Temporal Graph for Pedestrian Trajectory Pre-diction[J]. IEEE Access, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10634110)

- Yu C, Wang M. Pedestrian Trajectory Prediction Based on Multi-head Self-attention and Sparse Graph Convolution[C]//2024 4th International Conference on Neural Networks, Information and Communication (NNICE). IEEE, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10498338)

- Chen W, Sang H, Wang J, et al. IGGCN: Individual-guided graph convolution network for pedestrian trajectory prediction[J]. Digital Signal Processing, 2025. [paper](https://www.sciencedirect.com/science/article/abs/pii/S105120042400486X)

- Chen W, Sang H, Zhao Z. CWGCN: Cascaded Wavelet Graph Convolution Network for pedestrian trajectory prediction[J]. Computers and Electrical Engineering, 2025, 127: 110609. [paper](https://www.sciencedirect.com/science/article/abs/pii/S004579062500552X) [code](https://github.com/Chenwangxing/CWGCN)


 ### 3.3. Cross-spatial-temporal graph models
- Wu Y, Wang L, Zhou S, et al. Multi-stream representation learning for pedestrian trajectory prediction[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2023. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/25389) [code](https://github.com/YuxuanIAIR/MSRL-master)

- Chen W, Sang H, Wang J, et al. DSTIGCN: Deformable Spatial-Temporal Interaction Graph Convolution Network for Pedestrian Trajectory Prediction[J]. IEEE Transactions on Intelligent Transportation Systems, 2025. [paper](https://ieeexplore.ieee.org/abstract/document/10843981) [code](https://github.com/Chenwangxing/DSTIGCN_Master)

- Li R, Qiao T, Katsigiannis S, et al. Unified Spatial-Temporal Edge-Enhanced Graph Networks for Pedestrian Trajectory Prediction[J]. IEEE Transactions on Circuits and Systems for Video Technology, 2025. [paper](https://ieeexplore.ieee.org/abstract/document/10876405)


 ### 3.4. Scene graph models
- Cao D, Li J, Ma H, et al. Spectral temporal graph neural network for trajectory prediction[C]//2021 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9561461)

- Liu J, Li G, Mao X, et al. SparseGTN: Human Trajectory Forecasting with Sparsely Represented Scene and Incomplete Trajectories[C]//2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024: 13009-13016. [paper](https://ieeexplore.ieee.org/abstract/document/10801803)

- Zhu P, Zhao S, Deng H, et al. Attentive Radiate Graph for Pedestrian Trajectory Prediction in Disconnected Manifolds[J]. IEEE Transactions on Intelligent Transportation Systems, 2025. [paper](https://ieeexplore.ieee.org/abstract/document/10962257)


 ## 4. Heterogeneous graph-based methods
Heterogeneous graph-based methods incorporate multiple types of nodes (pedestrians, obstacles, vehicles) and edges (representing different kinds of relationships) into a unified heterogeneous graph. These methods support multimodal information fusion and heterogeneous interaction modeling. We further divide these methods into three categories based on the source of heterogeneity: **scene heterogeneous graph models**, **multi-agent type heterogeneous graph models**, and **comprehensive heterogeneous graph models**.

<img width="1000" alt="Figure 10" src="https://github.com/user-attachments/assets/50c91354-c6b9-479d-bf71-e51c7c572a04" />


 ### 4.1. Scene heterogeneous graph models
- Li J, Yang F, Tomizuka M, et al. Evolvegraph: Multi-agent trajectory prediction with dynamic relational reasoning[J]. Advances in neural information processing systems, 2020, 33: 19783-19794. [paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/e4d8163c7a068b65a64c89bd745ec360-Paper.pdf)

- Zou X, Sun B, Zhao D, et al. Multi-modal pedestrian trajectory prediction for edge agents based on spatial-temporal graph[J]. IEEE Access, 2020. [paper](https://ieeexplore.ieee.org/abstract/document/9082663) 

- Li J, Ma H, Zhang Z, et al. Spatio-temporal graph dual-attention network for multi-agent prediction and tracking[J]. IEEE Transactions on Intelligent Transportation Systems, 2021, 23(8): 10556-10569. [paper](https://ieeexplore.ieee.org/abstract/document/9491972)

- Zhao D, Li T, Zou X, et al. Multimodal Pedestrian Trajectory Prediction Based on Relative Interactive Spatial-Temporal Graph[J]. IEEE Access, 2022, 10: 88707-88718. [paper](https://ieeexplore.ieee.org/abstract/document/9862988)

- Yang X, Bingxian L, Xiangcheng W. SGAMTE-Net: A pedestrian trajectory prediction network based on spatiotemporal graph attention and multimodal trajectory endpoints[J]. Applied Intelligence, 2023, 53(24): 31165-31180. [paper](https://link.springer.com/article/10.1007/s10489-023-05132-z)


 ### 4.2. Multi-type agent heterogeneous graph models
- Eiffert S, Li K, Shan M, et al. Probabilistic crowd GAN: Multimodal pedestrian trajectory prediction using a graph vehicle-pedestrian attention network[J]. IEEE Robotics and Automation Letters, 2020. [paper](https://ieeexplore.ieee.org/abstract/document/9123560) 

- Zheng F, Wang L, Zhou S, et al. Unlimited neighborhood interaction for heterogeneous trajectory prediction[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021. [paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zheng_Unlimited_Neighborhood_Interaction_for_Heterogeneous_Trajectory_Prediction_ICCV_2021_paper.pdf)

- Alghodhaifi H, Lakshmanan S. Holistic Spatio-Temporal Graph Attention for Trajectory Prediction in Vehicle–Pedestrian Interactions[J]. Sensors, 2023, 23(17): 7361. [paper](https://www.mdpi.com/1424-8220/23/17/7361)

- Chen X, Zhang H, Hu Y, et al. VNAGT: Variational non-autoregressive graph transformer network for multi-agent trajectory prediction[J]. IEEE Transactions on Vehicular Technology, 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10121688)

- Zhou X, Chen X, Yang J. Edge-Enhanced Heterogeneous Graph Transformer With Priority-Based Feature Aggregation for Multi-Agent Trajectory Prediction[J]. IEEE Transactions on Intelligent Transportation Systems, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10807107)

- Sheng Z, Huang Z, Chen S. Ego‐planning‐guided multi‐graph convolutional network for heterogeneous agent trajectory prediction[J]. Computer‐Aided Civil and Infrastructure Engineering, 2024. [paper](https://onlinelibrary.wiley.com/doi/full/10.1111/mice.13301)

- Zhou X, Chen X, Yang J. Heterogeneous hypergraph transformer network with cross-modal future interaction for multi-agent trajectory prediction[J]. Engineering Applications of Artificial Intelligence, 2025. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0952197625001253) [code](https://github.com/zhou00NJUST/HHT-CFI)

- Li R, Katsigiannis S, Kim T K, et al. BP-SGCN: Behavioral Pseudo-Label Informed Sparse Graph Convolution Network for Pedestrian and Heterogeneous Trajectory Prediction[J]. IEEE Transactions on Neural Networks and Learning Systems, 2025. [paper](https://ieeexplore.ieee.org/abstract/document/10937142) [code](https://github.com/Carrotsniper/BP-SGCN)

- Ning N, Tian S, Li H, et al. Heterogeneous agents trajectory prediction with dynamic interaction relational reasoning[J]. Neurocomputing, 2025: 130543. [paper](https://www.sciencedirect.com/science/article/pii/S0925231225012159)


  ### 4.3. Comprehensive heterogeneous graph models
- Dong B, Liu H, Bai Y, et al. Multi-modal trajectory prediction for autonomous driving with semantic map and dynamic graph attention network[J]. arXiv preprint arXiv:2103.16273, 2021. [paper](https://arxiv.org/pdf/2103.16273)

- Li J, Ma H, Zhang Z, et al. Spatio-temporal graph dual-attention network for multi-agent prediction and tracking[J]. IEEE Transactions on Intelligent Transportation Systems, 2021, 23(8): 10556-10569. [paper](https://ieeexplore.ieee.org/abstract/document/9491972)

- Wang X, Yang X, Zhou D. Goal-CurveNet: A pedestrian trajectory prediction network using heterogeneous graph attention goal prediction and curve fitting[J]. Engineering Applications of Artificial Intelligence, 2024, 133: 108323. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0952197624004810)


 ## 5. High-order graph-based methods
High-order graph-based methods break through the limitation of pairwise modeling in conventional GNNs to explicitly capture group behavior and high-order interactions. High-order graph-based methods enhance the model’s ability to represent complex social dynamics in highly dense and interactive scenarios. According to the representation of high-order relationships, we further divide high-order graph-based methods into **high-order graph models** and **hypergraph models**.

<img width="1000" alt="Figure 11" src="https://github.com/user-attachments/assets/340b8e28-c8fa-444c-bc20-d5552988a313" />



 ### 5.1. High-order graph models
- Bae I, Jeon H G. Disentangled multi-relational graph convolutional network for pedestrian trajectory prediction[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2021. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/16174)
 
- Fang Y, Jin Z, Cui Z, et al. Modeling human–human interaction with attention-based high-order GCN for trajectory prediction[J]. The Visual Computer, 2022. [paper](https://link.springer.com/article/10.1007/s00371-021-02109-2)

- Wang K, Zou H. Social‐ATPGNN: Prediction of multi‐modal pedestrian trajectory of non‐homogeneous social interaction[J]. IET Computer Vision, 2024, 18(7): 907-921. [paper](https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/cvi2.12286)

- Kim S, Chi H, Lim H, et al. Higher-order Relational Reasoning for Pedestrian Trajectory Prediction[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024. [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_CVPR_2024_paper.pdf)

- Chen W, Sang H, Zhao Z. PCHGCN: Physically Constrained Higher-order Graph Convolutional Network for Pedestrian Trajectory Prediction[J]. IEEE Internet of Things Journal, 2025. [paper](https://ieeexplore.ieee.org/abstract/document/10948459) [code](https://github.com/Chenwangxing/PCHGCN-Master)



 ### 5.2. Hypergraph models

- Xu C, Li M, Ni Z, et al. Groupnet: Multiscale hypergraph neural networks for trajectory prediction with relational reasoning[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022. [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_GroupNet_Multiscale_Hypergraph_Neural_Networks_for_Trajectory_Prediction_With_Relational_CVPR_2022_paper.pdf) [code](https://github.com/MediaBrain-SJTU/GroupNet)

- Xu C, Wei Y, Tang B, et al. Dynamic-group-aware networks for multi-agent trajectory prediction with relational reasoning[J]. Neural Networks, 2024. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608023006299)

- Lee S, Lee J, Yu Y, et al. MART: MultiscAle Relational Transformer Networks for Multi-agent Trajectory Prediction[J]. arXiv preprint arXiv, 2024. [paper](https://arxiv.org/pdf/2407.21635) [code](https://github.com/gist-ailab/MART)

- Lin W, Zeng X, Pang C, et al. DyHGDAT: Dynamic Hypergraph Dual Attention Network for multi-agent trajectory prediction[C]//2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024. [paper](https://ieeexplore.ieee.org/abstract/document/10609870)

- Wang W, Mao L, Yang B, et al. Hyper-STTN: Social Group-aware Spatial-Temporal Transformer Network for Human Trajectory Prediction with Hypergraph Reasoning[J]. arXiv, 2024. [paper](https://arxiv.org/pdf/2401.06344)

- Chib P S, Nath A, Kabra P, et al. Ms-tip: Imputation aware pedestrian trajectory prediction[C]//International Conference on Machine Learning. PMLR, 2024: 8389-8402. [paper](https://openreview.net/forum?id=s4Hy0L4mml)

- Zhou X, Chen X, Yang J. Heterogeneous hypergraph transformer network with cross-modal future interaction for multi-agent trajectory prediction[J]. Engineering Applications of Artificial Intelligence, 2025. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0952197625001253) [code](https://github.com/zhou00NJUST/HHT-CFI)

- Lin X, Zhang Y, Wang S, et al. OST-HGCN: Optimized Spatial–Temporal Hypergraph Convolution Network for Trajectory Prediction[J]. IEEE Transactions on Intelligent Transportation Systems, 2025. [paper](https://ieeexplore.ieee.org/abstract/document/10857960)

- Hu Y, Chen X, Zhou Y, et al. A hypergraph-based dual-path multi-agent trajectory prediction model with topology inferring[J]. Engineering Applications of Artificial Intelligence, 2025, 152: 110799. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0952197625007997)



 ## Public Datasets
To comprehensively evaluate the accuracy, diversity, and distribution quality of predicted trajectories, researchers have proposed a variety of evaluation metrics.
1. UCY [dataset](https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data)

2. ETH [dataset](https://icu.ee.ethz.ch/research/datsets.html)

3. Stanford Drone Dataset (SDD) [dataset](https://cvgl.stanford.edu/projects/uav_data/)

4. NBA SportVU dataset (NBA) [dataset](https://github.com/linouk23/NBA-Player-Movements)

5. TrajNet++ [dataset](https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge)

6. NuScenes [dataset](https://www.nuscenes.org/)

7. Argoverse [dataset](https://www.argoverse.org/)

8. PIE [dataset](https://data.nvision2.eecs.yorku.ca/PIE_dataset/)

9. JAAD [dataset](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/)
   
<img width="1000" alt="微信图片_20250601155547" src="https://github.com/user-attachments/assets/5cd07fdc-58f6-419b-8219-f7f9748a2f62" />



 ## Evaluation Metrics
1. Average Displacement Error (ADE) [paper](https://ieeexplore.ieee.org/abstract/document/5459260)

2. Final Displacement Error (FDE) [paper](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8659.2007.01089.x)

3. Kernel Density Estimation (KDE) [paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Ivanovic_The_Trajectron_Probabilistic_Multi-Agent_Trajectory_Modeling_With_Dynamic_Spatiotemporal_Graphs_ICCV_2019_paper.html)

4. Average Mahalanobis Distance (AMD) [paper](https://link.springer.com/chapter/10.1007/978-3-031-20047-2_27)

5. Average Maximum Eigenvalue (AMV) [paper](https://link.springer.com/chapter/10.1007/978-3-031-20047-2_27)
   
<img width="1000" alt="微信图片_20250601155628" src="https://github.com/user-attachments/assets/7498835a-6804-4e85-bd26-430738724ebe" />






