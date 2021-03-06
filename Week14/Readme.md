# Group Members:
Gunjan Deotale, Abhijit Mali, Sanket Maheshwari, Sanjeev Raichur

Create this dataset and share a link to GDrive (publicly available to anyone) in this readme file. 
https://drive.google.com/drive/folders/1MST5DUffe3h9Q4B-x7tpNxXl4q4_E8ah

# Add your dataset statistics:

1 . Kinds of images (fg, bg, fg_bg, masks, depth)

fg :- Different Man, Woman, kids, group of person(for background transparency we have taken png images)
bg :- We restricted background to library images(for restricting size of image we have taken jpg images)
fg_bg :- bg superposed over fg (for restricting size of images we have taken jpg images)
masks :- masks extracted from fg images(we have taken grayscale images)(.jpg)
depth :- We have extracted depth images from fg_bg using nyu model(for restricing size of images we have taken grayscale
			images extracted from colormap)(.jpg) 
			
2. Total images of each kind
import os
print(sum([len(files) for r,d, files in os.walk('OverlayedMasks')]))
print(sum([len(files) for r,d, files in os.walk('DepthImage/')]))
print(sum([len(files) for r,d, files in os.walk('OverlayedImages')]))
print(sum([len(files) for r,d, files in os.walk('Foreground')]))
print(sum([len(files) for r,d, files in os.walk('ForegroundMask')]))
print(sum([len(files) for r,d, files in os.walk('flipForegroundMask')]))
print(sum([len(files) for r,d, files in os.walk('flipForeground')]))
print(sum([len(files) for r,d, files in os.walk('Background')]))
400000
400000
400000
100
100
100
100
100
	
3. The total size of the dataset :- 
!du -s OverlayedImages/
!du -s OverlayedMasks/
!du -s DepthImage/
!du -s Foreground/
!du -s ForegroundMask/
!du -s flipForeground/
!du -s flipForegroundMask/
!du -s Background/
!du -s ../Output/
6564640	OverlayedImages/
1143463	OverlayedMasks/
1635812	DepthImage/
857	Foreground/
177	ForegroundMask/
857	flipForeground/
176	flipForegroundMask/
6484	Background/
9547962	../Output/

4. Mean/STD values for your fg_bg, masks and depth images

fg_bg :- (BGR format) 
Mean: - [0.3234962448835791, 0.3776562499540454, 0.4548452917585805]
stdDev: - [0.22465676724491895, 0.2299902629415973, 0.23860387182601098]

masks :- (BGR format)
Mean: - [0.07863663756127236, 0.07863663756127236, 0.07863663756127236]
stdDev: - [0.2541994994472449, 0.2541994994472449, 0.2541994994472449]

depth :- (BGR format)
Mean: - [0.2943823440611593, 0.2943823440611593, 0.2943823440611593]
stdDev: - [0.15619204938398595, 0.15619204938398595, 0.15619204938398595]

# Show your dataset the way I have shown above in this readme
1. Background Images
![](Images/background.png)

2. Foreground Images
![](Images/foreground.png)

3. Foreground Masks
![](Images/Masks.png)

4. Foreground+Background
![](Images/OverlayedImages.png)

5. Foreground+Background Mask
![](Images/OverlayedDepthMask.png)

6. DepthMap
![](Images/Overlayed.png)

# Explain how you created your dataset
1. how were fg created with transparency :- 
	We mainly downloaded images from internet without background, for some images we extracted foreground
    by using background removal technique in PowerPoint as shown in lecture.
	
2. how were masks created for fgs
	We figure out that mask images are nothing but alpha channels of images. So we extracted masks using following code
	image = cv2.imread("Foregroundimg.png", cv2.IMREAD_UNCHANGED)
	imagealpha = image[:,:,3]
	cv2.imwrite("ForegroundMask.jpg", imagealpha)
	
3. how did you overlay the fg over bg and created 20 variants
    1. first all background images were resized to 160x160
	2. all foreground images were resized to 80(max side) and other side was reshaped as per aspect ratio
	3. images were randomly placed by choosing starting x,y randomly on background, but also making sure that foreground
	   image doesnot go out of background image.
	4. Code for generation of data is mentioned in DataGeneration.py
	
4. how did you create your depth images? 
	1. Although we used mostly same code as given in assignment for depthimage, we need to modify code to save images.
	2. We have modified code in utils.py and test.py to save images directly to drive
	3. Following is code for saving data
	https://github.com/gdeotale/EVA4/tree/master/Week14/DenseDepth/test.py
	https://github.com/gdeotale/EVA4/tree/master/Week14/DenseDepth/utils.py
	
5. how full data was created?
	1. Creating full dataset singlehandedly was quite taxing, so we subdivided data in 4 people, each one created 100000 fg_bg, 100000 masks, 100000 depth images
	2. At the end we merged all folders data in one drive by sharing of folders with each other.
	
