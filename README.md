# Golem Neural Network Image Aesthetic Assessment

Machine learning and deep learning requires computing power. Upgrading hardware or using cloud computing could both cost an arm and a leg. 
By using golem, sharing CPU copmuting power, AI training could be possible for everyone.
(GPL Version-3 license)
# 0. Intro Video and Installation

- Click to watch the intro video

[![Watch the video](https://img.youtube.com/vi/pnGtBH0EGaE/hqdefault.jpg)](https://youtu.be/pnGtBH0EGaE)


Step 1:
Follow Golem Tutorial
```
yagna service run
yagna payment init -r
python3 -m venv ~/.envs/yagna-python-tutorial
source ~/.envs/yagna-python-tutorial/bin/activate
export YAGNA_APPKEY=insert-your-32-char-app-key-here
```
Step 2:
Run Frontend
```
npm install
npm run start
```
Step 3:
open your browser and go to:
http://localhost:3000/

Note: 
Neural Network Image Aesthetic Assessment Pytorch Model weights file:
https://drive.google.com/file/d/1esdd-6SyB9vhDSvUgh0VUBVWpzq0ATty/view?usp=sharing

# 1. Introduction

Aesthetic assessment quantifies semantic level characteristics associated with emotions and beauty in images. The AVA dataset[1] contains about 255,000 images, rated based on aesthetic qualities by amateur photographers. Each photo is scored by an average of 200 people in response to photography contests. The image ratings range from 1 to 10, with 10 being the highest aesthetic score associated to an image. Aesthetic quality of a photograph can be represented by the mean score, and unconventionality of it closely correlates to the score deviation(high score variance tend to be subject to interpretation, low score variance seem to represent conventional styles).


![GolemAestheticAssessment](/imgs/F1.png)
- Figure 1. Colors of the image are not highly correlated with the Aesthetic assessment of an image.


# 2. Paper Survey


![GolemAestheticAssessment](/imgs/F2_2.png)

Most of the aesthetic assessment models we surveyed only divide images into two classes (Good or Bad). This kind of binary classification only focuses on the average score prediction discard the prediction of the score distribution.



![GolemAestheticAssessment](/imgs/F2_1.png)
- Figure 2. The drawbacks of the binary classification problem




# 3. Google NIMA

Google’s NIMA[2] is the state-of-the-art in neural network image assessment field. In Google’s NIMA, the neural network structure is quite simple. They use baseline image classifier network(such as ResNet, R-FCN,......) as their image characteristics capturing layer. The output of the characteristics capturing layer is fully-connected and input to the EMD (Earth Movers Distance) Loss layer. The prediction can be made using the output of the EMD Loss layer. Figure 2 illustrates the idea of the work flow of Google’s Nima neural network model. 


![GolemAestheticAssessment](/imgs/F3_1.png)
- Figure 3. The neural network architecture of NIMA Aesthetic assessment Model.

EMD is defined as the minimum cost to move the mass of one distribution to another. Given the ground truth and estimated probability mass functions p and pˆ , with N ordered classes of distance ∥si − sj ∥r , the normalized EMD [3] can be expressed as: 

![GolemAestheticAssessment](/imgs/F3_2.jpg)

# 4. Results


![GolemAestheticAssessment](/imgs/F4.png)

- Figure 4.  Scores of different images using NIMA Aesthetic assessment  Model.

# 5. Reference

[1] N. Murray, L. Marchesotti, and F. Perronnin, AVA: A large-scale database for aesthetic visual analysis, in Computer Vision and Pattern Recognition(CVPR),2012IEEEConference on. IEEE,2012,pp.2408 2415. 1,2,3,7,8,9,10

[2] H. Talebi and P. Milanfar. Nima: Neural image assess- ment. TIP, 2018. 3

[3] E. Levina and P. Bickel, The earth movers distance is the Mallows distance: Some insights from statistics, in Com- puter Vision, 2001. ICCV 2001. Proceedings. Eighth IEEE International Conference on, vol. 2. IEEE, 2001, pp. 251256. 6

# 6. Future Work

Federated Learning(Privacy Preserved Distributed Deep Learning) Blockchain + AI: 
 - Motivation: Privacy and Decentralization are the next big thing in the Tech field. EU announced GDPR. US announced Federal information privacy laws. Facebook was fined $US5 billion for Cambridge Analytica privacy violations. How to train the useful deep learning model without violate user's privacy will be very important. Golem Layer2 can definitedly play do federated learning and reward users who join training process with tokens. 


Gaming: 
 - Golem Neural Network Image Aesthetic Assessment can become a platform. Everyone can share good looking pictures on this platform. We can also issue NFT for the high score picture. We can also hold photography competition on the platform and reward participants using tokens.

