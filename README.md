# **Advanced Lane Finding Project** 

## Fourth and final project for Udacity's Self Driving Car Nanodegree - Term 1

### This is the project writeup. [Instructions](Instructions.md) are here. Project's expectations are called rubic points and they are [here](https://review.udacity.com/#!/rubrics/571/view).

---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image01]: ./examples/objectpoints.png "Points"
[image02]: ./examples/undistorted.png "Undistorted"
[image03]: ./examples/perspective.png "Perspective Transformation"
[image04]: ./examples/thresholds.png "Thresholds"
[image05]: ./examples/draw_lane.png "draw_lane()"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "P4_pipeline.ipynb".   

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

![alt text][image01]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image02]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. (Combined threshold function in a code cell with "combined_thresh()" function of the IPython notebook). I try to test all of methods that I learned in the class, such as magnitude, direction, RGB and HLS channel. I chose that the following methods and thresholds due to work well.

* The magnitude, with kernel size of 9, a min threshold of 15, and a max threshold of 70.
* The H channel from HLS color space, with a min threshold of 19 and a max threshold of 60. I expected that the intersection of magnitude and H channel would classify yellow lane lines with shadows on the road.
* The R channel from RGB color space, with a min threshold of 230 and a max threshold of 255. Yellow lane lines are clearer with that.
* The B channel from RGB color space, with a min threshold of 200 and a max threshold of 250.
* The L channel from HLS color space, with a min threshold of 220 and a max threshold of 255. White lane lines are clearer with that.

So, combined method has - 

```
mag_and_h_channel[(magnitude == 1) & (h_channel == 1)] = 1
combined[(mag_and_h_channel == 1) | (r_channel == 1) | (b_channel == 1) | (l_channel == 1)] = 1
```


I used a combination of color and gradient thresholds to generate a binary image (Combined threshold function in a code cell with "combined_thresh()" function of the IPython notebook). Here's an example of my output for this step. 

![alt text][image04]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in 6th code cell of the IPython notebook.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python

	offset=30, corx1=0.38, corx2=0.63, cory=0.67

    src = np.float32([[img_size[0]*corx1, img_size[1]*cory],
                      [img_size[0]*corx2, img_size[1]*cory], 
                      [img_size[0]-offset, img_size[1]-offset], 
                      [offset, img_size[1]-offset]])
    
    dst = np.float32([[offset, offset], 
                      [img_size[0]-offset, offset], 
                      [img_size[0]-offset, img_size[1]-offset], 
                      [offset, img_size[1]-offset]])

```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 486, 482		| 30, 30		|
| 806, 482		| 1250, 30		|
| 1250, 690		| 1250, 690		|
| 30, 690		| 30, 690		|

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image03]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Next step was to fit polynomial between left and right lane lines. 

* I checked peaks in a histogram of the image to determine location of both lane lines
* I processed all non zero pixels around histogram peaks (8th last code cell from the bottom in IPython notebook ). 
* I attempted to fit a polynomial to left and right lanes using by np.polyfit() (7th last code cell from the bottom in IPython notebook with following code).

```
position = (rightx_int+leftx_int)/2
distance_from_center = abs((640 - position)*3.7/800) 
```

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I implemented this in 8th last code cell from the bottom in IPython notebook. Here is the code.

```
    def radius_of_curvature(self, xvals, yvals):
        # meters per pixel in y dimension
        ym_per_pix = 15./720
        # meteres per pixel in x dimension
        xm_per_pix = 3.7/800
        y_eval = np.max(yvals)
        fit_cr = np.polyfit(yvals*ym_per_pix, xvals*xm_per_pix, 2)
        curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        return curverad
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this in 7th last code cell from the bottom in IPython notebook in the function `draw_lane()`.  Here is an example of my result on a test image:

![alt text][image05]

For video, in order to show the result smoothly, I setted up to remember the average of coefficients of previous 10 steps' polynomial.

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](project_video_output.mp4)
Here's a [link to challenge video result](challenge_video_output.mp4)
Here's a [link to harder challenge video result](harder_challenge_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This pipeline doesn't work on challenge and harder challenge videos. I should consider - 

* Other binary thresholds with color channel (For an example - LUV or Lab)
* Drawing lane lines when turn is very sharp (left or right)

