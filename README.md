# facial_expression_RandomForest
基于 随机森林算法 和 FER_2013表情数据集 的人脸表情识别系统
大纲:
1.数据处理部分：
1.图像质量预处理
首先先针对fer_2013原始数据集去进行训练，会发现在我们使用的cascade级联人脸识识别器对人脸进行特征点标注时会出现绝大多数人脸都无法识别成功的情况，eg:2000/20000,在对原始的数据集进行观察之后发现原始数据集：1.像素数过小，导致图像模糊2.图像黑白对比度不明显，一些图像像素分布不均匀。需要进行图像处理技术来解决这个问题，同时实现在检测时期的鲁棒性(头的转向，歪头等等情况，因为数据集也存在这类表情图像)
我们使用了水平反转将图像左右翻转，用以模拟不同人脸朝向下的情况，同时为应对歪头的情况，我们还使用了图像旋转技术将人脸图像随机旋转-15——+15度。
同时我们还考虑到在进行人脸识别时，会存在人脸远近的问题，我们不想在识别同一张面孔的时候因为距离远近而导致识别出错，所以我们使用随机对图像放缩的方式来规避这个问题，我们还观察到一些图像会存在亮度过暗的状态，所以我们也对全体数据集做了亮度的提升
2.图像尺寸处理
为了解决像素数不够大的情况，我们使用超分辨率技术将48*48的像素的图像转换为了192*192像素的图像，我们直接使用opencv的超分辨率模块就可以做到。有趣的是顺序的问题，如果在像素数扩大之后在进行图像的处理（图像增强）会导致图片变模糊效果并不好，而将顺序反过来则不会。下面我们可以看到完整图像处理步骤过后的效果图
我们可以很明显观察到图像的变化与增强还是很有效果的
<img width="108" height="100" alt="image" src="https://github.com/user-attachments/assets/e77cce9c-d8ee-46e0-a676-4d8a15d44b83" />
<img width="100" height="100" alt="image" src="https://github.com/user-attachments/assets/52660d61-0c67-48be-bfdb-fb94fbd234c5" />
<img width="96" height="100" alt="image" src="https://github.com/user-attachments/assets/cae801ca-c141-4ff9-8d54-0c0f6d5c8c68" />
<img width="99" height="99" alt="image" src="https://github.com/user-attachments/assets/6abaccb1-b90d-4a23-bfda-49cf4ed70435" />
<img width="36" height="36" alt="image" src="https://github.com/user-attachments/assets/09f22e99-025c-427d-87cf-dbcbf3e8730f" />
<img width="36" height="36" alt="image" src="https://github.com/user-attachments/assets/72177fee-b982-4ef5-87a0-2ad22ce696cf" />
<img width="36" height="36" alt="image" src="https://github.com/user-attachments/assets/19c0841f-b06a-434b-abb4-aa2cf5a0110d" />
<img width="36" height="36" alt="image" src="https://github.com/user-attachments/assets/4ac616cf-966a-4f5d-af5d-2f7e7abcae34" />

随后我们使用dlib_cnn去进行人脸识别和特征点标注。效果对比如下：
mediapipe<img width="96" height="96" alt="image" src="https://github.com/user-attachments/assets/c8c9dd72-8424-4c83-8809-609098a9c0c9" />

<img width="96" height="95" alt="image" src="https://github.com/user-attachments/assets/0c04648e-38b8-4097-b8b0-0f1e44d20fae" />
<img width="96" height="95" alt="image" src="https://github.com/user-attachments/assets/6d37070e-c701-4051-8a50-75341401b3dc" />
<img width="95" height="89" alt="image" src="https://github.com/user-attachments/assets/81fb155b-fabf-4661-b03b-f949d28dc668" />
dlib----------->我们可以看图像增强之前的标点：
<img width="95" height="92" alt="image" src="https://github.com/user-attachments/assets/f1dbe6df-9295-4da2-8a41-7ae0548a5b5d" />

3.模型训练部分：
Expert_Level 模型训练部分
我们采用随机森林分类器去实现表情的分类
1、首先，仅对Dlib提取出来的68个特征点做中心化和归一化。模型的表现如下：
<img width="245" height="142" alt="image" src="https://github.com/user-attachments/assets/3d223093-f887-4e65-9ae2-1679591c6988" />
2、分析Dlib的68特征点，我们认为有一些点对于表情的判别并没有太多参考价值（比如刻画面部轮廓的点）。所以我们只保留了眼睛、鼻子、嘴巴等关键特征点，重新进行训练，得到如下结果：
<img width="256" height="147" alt="image" src="https://github.com/user-attachments/assets/e21809dd-57f2-46b0-ac3a-bf451cff3d64" />
<img width="256" height="192" alt="image" src="https://github.com/user-attachments/assets/bb9c8c5e-8026-40f0-902d-0f6dff8adf56" />
由图看出，总的正确率相比之前有了轻微的提升。
进一步观察混淆矩阵，我们发现不管是什么表情，都有比较大的倾向被判定为neutral。
我们认为，这是由于我们没有充分提取面部的特征点，所以导致分类器把一些本不该为neutral的表情变成了neutral。
所以，我们通过特征工程，从面部额外提取了一些指标。具体如下所示：
特征编号	名称	含义	相关情绪例子
1	eyebrow_length	眉毛长度	愤怒、惊讶
2	nose_length	鼻梁长短	个体差异
3	mouth_width	嘴巴宽度	微笑、厌恶
4	mouth_height	嘴巴高（上下）	惊讶、恐惧
5	eye_width	眼睛水平宽度	-
6	eye_height	眼睛垂直高度	睁眼、瞪眼
7	left_eyebrow_curve	眉毛中点偏差	愤怒、疑惑
8	right_eyebrow_curve	同上	愤怒、惊讶
9	inner_mouth_height	内嘴张开程度	惊讶、恐惧、大笑
10	nose_width	鼻翼宽度	愤怒、厌恶
11	left_eye_openness	左眼开合	疲倦、眨眼
12	right_eye_openness	右眼开合	疲倦、瞪眼
注：眼睛垂直高度是眼睛在图像上最上方的点和最下方的点的y坐标之差，而开合程度是描述瞳孔张开的程度

同时，我们考虑到，原先我们提取出的点是41个，将二维矩阵平面化成一维向量后有82多个点。
（1）原先识别效果不好的一些表情，正确率有了一定的提升，而happy和neutral正确率有了一定的下降。然而，他们原先的正确率就很高了，现在仍然保持着较高的正确率。我们认为这是有意义的。
（2）被误判为neutral的样本数量减少了，这说明分类器获取到了更多表情的特征，不再那么倾向于认为是neutral
<img width="169" height="182" alt="image" src="https://github.com/user-attachments/assets/8f581c7d-da18-4e72-945c-8805b4280084" />
<img width="256" height="170" alt="image" src="https://github.com/user-attachments/assets/5b88a466-5fcc-41d6-ac7d-788d4bfa2b2f" />
<img width="256" height="199" alt="image" src="https://github.com/user-attachments/assets/37499643-eb72-4604-987e-ff16c14ea21a" />
我们可以看到其实对比上一张图来说，我们可以发现其实happy和neutral这样的识别准确数会有轻微下降，但对于sad\disgust这样的难区分表情其实是有大的提升的，这对我们来说也是一个很大的进步
3、随后，经过调整randomforest分类器的参数（主要是增加深度），我们能够将accuracy提高到0.58左右。
<img width="227" height="130" alt="image" src="https://github.com/user-attachments/assets/42fb13ec-692b-4106-a35a-d1c79f3f9345" />
<img width="227" height="171" alt="image" src="https://github.com/user-attachments/assets/5c84d77a-86d4-40b7-9fc4-f619c8702ac4" />











4、我们还尝试了用XGBoost来分类，分类效果比随机森林略好。

<img width="227" height="141" alt="image" src="https://github.com/user-attachments/assets/f559b669-b14e-4afd-8dae-89611e56e3cc" />
<img width="227" height="168" alt="image" src="https://github.com/user-attachments/assets/118c2aec-7fb5-47eb-beb0-1d7316f540a4" />

我们认为原因可能如下：随机森林属于 Bagging 模型，它通过构建多棵相互独立的决策树并进行多数投票来进行分类，因此更擅长处理具有强信号的特征。然而，它对于复杂交互特征和弱信号的区分能力较弱，容易在模糊样本上做出不稳定的决策。相比之下，XGBoost 采用 Boosting 策略，每一棵树都在试图“修正”前一棵树的错误预测，这使得它更擅长在复杂边界和弱特征之间挖掘潜在的区分模式。

4.特效添加部分：
<img width="154" height="125" alt="image" src="https://github.com/user-attachments/assets/b1e5c5ef-2ac5-4188-9949-5b856c81559f" />
<img width="158" height="126" alt="image" src="https://github.com/user-attachments/assets/e8b0d5df-5894-4f89-b7f2-cd88e5e5d093" />
<img width="151" height="120" alt="image" src="https://github.com/user-attachments/assets/86381a73-13b1-489b-995c-9e1f306b8597" />
<img width="159" height="126" alt="image" src="https://github.com/user-attachments/assets/698c93f7-df7d-4976-99eb-fbf5947274ee" />
我使用一个12个帧的队列，当队列满12帧时，帧会自动溢出，我们选用8帧为一个门槛，如果超过门槛，会自动判别为一个特效，此时我们加载所有特效。另外使用图片叠加技术去实现emoji图片添加，以及滤镜添加（这个其实是通过webcam相关的api直接调用）










我使用一个12个帧的队列，当队列满12帧时，帧会自动溢出，我们选用8帧为一个门槛，如果超过门槛，会自动判别为一个特效，此时我们加载所有特效。另外使用图片叠加技术
