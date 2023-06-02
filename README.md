# Segmentation-Method-for-Intracerebral-Hemorrhage
<p align="center">
  <img src="https://i.postimg.cc/0Qqg89yD/portada.jpg"> 
</p>

The presence of intracerebral hemorrhage (ICH) poses significant challenges in medical practice, with a high mortality rate and potential irreversible damage. Current treatments focus on stopping bleeding, removing clots, and relieving brain pressure. Computed tomography (CT) imaging plays a crucial role in accurate diagnosis and treatment planning.

To address these challenges, developing a CT image segmentation algorithm holds great promise. This algorithm aims to improve efficiency, accuracy, and consistency in identifying and delineating hemorrhages, benefiting treatment planning and medical education. The primary objective of this Bachelor's Thesis (TFG) is to develop a convolutional neural network (CNN) based algorithm for cerebral hemorrhage segmentation, leveraging the success of deep learning techniques in image analysis tasks.

The research will utilize the INSTANCE 2022 database, comprising 100 CT images of cerebral hemorrhages with medical annotations. These annotations will enable the evaluation of the algorithm's automatic segmentation results using overlap measures. The algorithm will be implemented in Python, utilizing the "Tensorflow" deep learning library.

Overall, this research project aims to significantly impact medical imaging and intracerebral hemorrhage management. By developing an advanced segmentation algorithm, it seeks to improve treatment decision-making, patient outcomes, and the quality of care provided.

In recent years, advancements in medical imaging and artificial intelligence (AI) techniques have propelled the evolution of ICH detection and segmentation. AI-based approaches, particularly CNNs, have shown promise in automating ICH detection from CT and MRI scans. Researchers strive to enhance model accuracy, incorporate explainability and interpretability, ensure generalizability across diverse patient populations, and enable real-time analysis on resource-constrained devices. Integrating these models into clinical workflows aims to enhance patient care and outcomes in cases of intracranial hemorrhage.


The dataset for this project consisted of 100 3D CT images of patients with intracranial hemorrhages. To optimize training time and focus on images with significant lesions, the dataset was preprocessed to extract and save only the slices containing substantial lesions, resulting in 95 usable images. The dataset was divided into 80% training (78 patients) and 20% testing (17 patients) subsets. During training, a validation step was performed using the included validation dataset.

Preprocessing involved visualizing the images and masks using ITK-Snap software to identify the intensity ranges corresponding to the skull, brain, and lesions. The skull was eliminated, brain intensity was adjusted, and lesion intensity was enhanced to improve lesion detection. The volumes were converted into slices, and only slices with sufficiently large lesions were saved, excluding images without lesions or with small lesions that could introduce confusion during training.

<p align="center">
  <img src="https://i.postimg.cc/B6HNqm5t/image-TFG-1.jpg">
  <br>
  <sub>Figure 1: Comparison between the default image and a manual contrast correction using Mango</sub>
</p>


For model training, a 2D U-Net architecture was employed, and the dataset was further divided into training and testing patients. A total of 563 training images with significant lesions were used. The 'Adam' optimizer was chosen, as it combines adaptive learning rates and momentum-based optimization to handle different gradients and accelerate learning in complex models like U-Net. The binary cross-entropy loss function was used, which is commonly employed for image segmentation tasks with U-Net models.

Validation metrics such as accuracy and Dice coefficient were used to evaluate the model's performance during training. The model was tested on the 17 patients' 3D images, which were converted into 2D slices for input. The 3D images were reconstructed using the predicted slices, and evaluation metrics such as Dice coefficient and false positives were calculated between the ground truth and predicted volumes for each patient. The Dice coefficient measures the similarity between two images, while false positives assess the number of incorrectly predicted regions per image, contributing to the analysis of the model's performance.
<p align="center">
  <img src="https://i.postimg.cc/jjFgZWcQ/Image-TFG2.jpg">
  <br>
  <sub>Figure 2: Bar graph representing the Dice coefficient of the final segmentation of each patient in the test</sub>
</p>
Figure 2: Bar graph representing the Dice coefficient of the final segmentation of each patient in the test

The evaluation of the algorithm showed a Dice coefficient of 0.5482 ± 0.2740, positioning it at 284, and a false positives count of 12.5 ± 7.35. However, excluding patients 13 and 15 from the evaluation resulted in an improved Dice coefficient of 0.6148 ± 0.2128, placing the algorithm at position 266.
<p align="center">
  <img src="https://i.postimg.cc/Qxd6Gkzk/imatge-TFG3.jpg"> 
  <br>
  <sub>Figure 3: Example of segmentation.</sub>
</p>

<p align="center">
  <img src="https://i.postimg.cc/T1Myx6f5/Imatge-TFG4.jpg">
  <br>
  <sub>Figure 4: Segmentation reconstruction visualization with ITK-Snap of a good segmentation.</sub>
</p>

A challenge encountered in my UNET model was the imbalanced class distribution between instances of intracranial hemorrhage (ICH) and non-ICH cases. The rarity of ICH instances led to a bias towards the majority class, resulting in a higher number of false positives. To address this, it is acknowledged that incorporating non-ICH images in the training data could have improved the model's accuracy. By achieving a more balanced representation of both classes, the model would have had the opportunity to learn more effectively and potentially achieve better performance. In testing, when the model was evaluated solely on ICH images, the accuracy reached approximately 60%.

Furthermore, a limitation observed in the UNET model architecture utilized was the limited capture of global or long-range dependencies. The encoder-decoder structure of UNET has a restricted receptive field, which can hinder the model's ability to capture relationships between distant regions in the image. This contextual information is crucial for accurate segmentation, particularly in understanding the relationships between different parts of the brain in the presence of ICH. To overcome this limitation, it is recognized that a 3D U-net architecture could be beneficial. By incorporating the additional dimension of depth, a 3D U-net would capture spatial dependencies more effectively and provide a better understanding of the entire volume of brain images. This approach would likely enhance the model's ability to capture global or long-range dependencies, leading to improved segmentation accuracy for ICH lesions.

The trained U-Net model showed moderate performance for large, circular, and centrally located lesions but struggled with small, elongated, and non-centrally located lesions. To overcome this limitation, post-processing techniques can be employed to refine and contour the segmentation results, thereby improving accuracy and precision.

These findings highlight the need for further research and improvements in accurately detecting ICH lesions using the proposed approach. Enhancing the model's performance, especially in identifying smaller and non-centrally located lesions, requires exploring alternative techniques, modifications to the preprocessing step, and advanced algorithms for lesion detection.

Possible improvements include adopting a 3D U-net architecture to incorporate volumetric information and capture spatial dependencies between slices. This would enhance lesion detection and reduce false positives, resulting in improved segmentation performance.

In terms of preprocessing, personalizing the intensity range for each patient based on their individual characteristics could improve lesion visibility. Normalizing the intensity range by mapping pixel intensities according to the mean intensity of each patient's brain tissue would enhance lesion detection.

Furthermore, incorporating post-processing techniques such as morphological operations and connected component analysis can refine the segmentation results and eliminate spurious detections, leading to more accurate segmentation.

In conclusion, implementing a 3D U-net architecture, personalizing intensity ranges during preprocessing, and incorporating post-processing techniques can significantly improve the accuracy and robustness of the ICH segmentation model. These enhancements address major limitations and challenges, paving the way for more precise and reliable results.

