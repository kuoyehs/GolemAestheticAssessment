# Neural Network Image Aesthetic Assessment

Machine learning and deep learning requires computing power. Upgrading hardware or using cloud computing could both cost an arm and a leg. 
By using golem, sharing CPU copmuting power, AI training could be possible for everyone.

# 1. Introduction

Aesthetic assessment quantifies semantic level characteristics associated with emotions and beauty in images. The AVA dataset[1] contains about 255,000 images, rated based on aesthetic qualities by amateur photographers. Each photo is scored by an average of 200 people in response to photography contests. The image ratings range from 1 to 10, with 10 being the highest aesthetic score associated to an image. Aesthetic quality of a photograph can be represented by the mean score, and unconventionality of it closely correlates to the score deviation(high score variance tend to be subject to interpretation, low score variance seem to represent conventional styles).

Figure 1. Colors of the image are not highly correlated with the Aesthetic assessment of an image.

# 2. Paper Survey


![GolemAestheticAssessment](/imgs/F2_1.png)

Most of the aesthetic assessment models we surveyed only divide images into two classes (Good or Bad). This kind of binary classification only focuses on the average score prediction discard the prediction of the score distribution.

![GolemAestheticAssessment](/imgs/F2_2.png)
- Figure 2. The drawbacks of the binary classification problem




# 3. Google NIMA

Google’s NIMA[2] is the state-of-the-art in neural network image assessment field. In Google’s NIMA, the neural network structure is quite simple. They use baseline image classifier network(such as ResNet, R-FCN,......) as their image characteristics capturing layer. The output of the characteristics capturing layer is fully-connected and input to the EMD (Earth Movers Distance) Loss layer. The prediction can be made using the output of the EMD Loss layer. Figure 2 illustrates the idea of the work flow of Google’s Nima neural network model. 

![GolemAestheticAssessment](/imgs/F3_1.png)
- Figure 3. The neural network architecture of NIMA Aesthetic assessment  Model.



EMD is defined as the minimum cost to move the mass of one distribution to another. Given the ground truth and estimated probability mass functions p and pˆ , with N ordered classes of distance ∥si − sj ∥r , the normalized EMD [3] can be expressed as: 

# 4. Results
