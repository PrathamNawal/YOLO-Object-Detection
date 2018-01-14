<h1>YOLO Object Detection</h1>
<hr />
<p>The objective of this project is to create a pipeline (model) to draw bounding boxes around cars in a video. This video was taken from a camera mounted on the front of a car.</p>
<h3>I. Histogram of Oriented Gradients (HOG)</h3>
<h4><a id="user-content-1-extract-hog-features-from-the-training-images" class="anchor" href="https://github.com/jessicayung/self-driving-car-nd/tree/master/p5-vehicle-detection#1-extract-hog-features-from-the-training-images" aria-hidden="true"></a>1. Extract HOG features from the training images.</h4>
<ol>
<li>
<p>Read in all&nbsp;<code>vehicle</code>&nbsp;and&nbsp;<code>non-vehicle</code>&nbsp;images. Here are two example images, the first from the&nbsp;<code>vehicle</code>&nbsp;class and the second from the&nbsp;<code>non-vehicle</code>&nbsp;class:</p>
<ul>
<li><a href="https://github.com/jessicayung/self-driving-car-nd/blob/master/p5-vehicle-detection/readme_images/vehicle-example9.png" target="_blank" rel="noopener"><img src="https://github.com/jessicayung/self-driving-car-nd/raw/master/p5-vehicle-detection/readme_images/vehicle-example9.png" alt="Vehicle" /></a></li>
<li><a href="https://github.com/jessicayung/self-driving-car-nd/blob/master/p5-vehicle-detection/readme_images/non-vehicle-example1.png" target="_blank" rel="noopener"><img src="https://github.com/jessicayung/self-driving-car-nd/raw/master/p5-vehicle-detection/readme_images/non-vehicle-example1.png" alt="Non-vehicle" /></a></li>
</ul>
</li>
<li>
<p>Use&nbsp;<code>skimage.feature.hog(training_image, [parameters=parameter_values])</code>&nbsp;to extract HOG features and HOG visualisation.</p>
<ul>
<li>Wrapped in function&nbsp;<code>get_hog_features</code>.</li>
</ul>
</li>
</ol>
<p>Code in Section 1 of&nbsp;<code>helperfunctions.py</code>. Relevant functions:&nbsp;<code>get_hog_features</code>&nbsp;and&nbsp;<code>extract_features</code>.</p>
<h4><a id="user-content-2-choose-hog-parameters" class="anchor" href="https://github.com/jessicayung/self-driving-car-nd/tree/master/p5-vehicle-detection#2-choose-hog-parameters" aria-hidden="true"></a>2. Choose HOG parameters.</h4>
<ul>
<li>I wanted to optimise for HOG parameters systematically, so I wrote a script&nbsp;<code>hog_experiment.py</code>&nbsp;that enables me to easily run through different HOG parameters and save the classifier accuracy, the HOG visualisation (image) and bounding boxes overlaid on the video frame (image).
<ul>
<li>I later used&nbsp;<code>p5-for-tuning-classifier-parameters.ipynb</code>.</li>
<li>I used a spreadsheet to note down the parameters used each time, the accuracy, training time and the quality of the bounding boxes for each of the six test images. (E.g. was each car detected? If so, how well was it covered by the bounding boxes (how many bounding boxes + did it cover the car in full or only partially?) How many false positives were there?)</li>
</ul>
</li>
<li>I then picked the HOG parameters based on classifier accuracies and looking at the output images. I would like to make this process more rigorous instead of sort of basing it on intuition.
<ul>
<li>It was difficult to do this because the classifier accuracy was usually above 99% and was often shown as 1.0 even if the classifier later drew many false positive bounding boxes.</li>
<li>I later realised that the accuracy had been so high because I'd only been using 500 images from each category (vehicles and non-vehicles). But I still couldn't rely only on the classifier accuracy as a measure because a higher accuracy didn't always give me 'better' bounding boxes.</li>
</ul>
</li>
</ul>
<p>Code in second cell in Section 1.2.</p>
<h4><a id="user-content-3-train-a-classifier-using-selected-hog-features-and-colour-features" class="anchor" href="https://github.com/jessicayung/self-driving-car-nd/tree/master/p5-vehicle-detection#3-train-a-classifier-using-selected-hog-features-and-colour-features" aria-hidden="true"></a>3. Train a classifier using selected HOG features and colour features.</h4>
<ol>
<li>Format features using&nbsp;<code>np.vstack</code>&nbsp;and&nbsp;<code>StandardScaler()</code>.</li>
<li>Split data into shuffled training and test sets</li>
<li>Train linear SVM using&nbsp;<code>sklearn.svm.LinearSVC()</code>.</li>
</ol>
<p>&nbsp;</p>
<h3>A Tiny Model</h3>
<p>The&nbsp;<code>yolo-tiny.cfg</code>&nbsp;is based on the Darknet&nbsp;<a href="http://pjreddie.com/darknet/imagenet/#reference">reference network</a>. You should already have the config file in the&nbsp;<code>cfg/</code>&nbsp;subdirectory. Download the pretrained weights&nbsp;<a href="http://pjreddie.com/media/files/yolo-tiny.weights">here (172 MB)</a>. Then you can run the model!</p>
<pre><code>./darknet yolo test cfg/yolo-tiny.cfg yolo-tiny.weights
</code></pre>
<p>The tiny version of YOLO only uses 611 MB of GPU memory and it runs at more than 150 fps on a Titan X.</p>
<h3><a id="user-content-yolo-model-comparison" class="anchor" href="https://github.com/pjreddie/darknet/wiki/YOLO:-Real-Time-Object-Detection#yolo-model-comparison" aria-hidden="true"></a>YOLO Model Comparison</h3>
<ul>
<li><code>yolo.cfg</code>&nbsp;is based on the&nbsp;<code>extraction</code>&nbsp;network. It processes images at 45 fps, here are weight files for&nbsp;<code>yolo.cfg</code>&nbsp;trained on&nbsp;<a href="http://pjreddie.com/media/files/yolo.weights">2007 train/val+ 2012 train/val</a>, and trained on&nbsp;<a href="http://pjreddie.com/media/files/yolo-all.weights">all 2007 and 2012 data</a>.</li>
<li><code>yolo-small.cfg</code>&nbsp;has smaller fully connected layers so it uses far less memory. It processes images at 50 fps, here are weight files for&nbsp;<code>yolo-small.cfg</code>&nbsp;trained on&nbsp;<a href="http://pjreddie.com/media/files/yolo-small.weights">2007 train/val+ 2012 train/val</a>.</li>
<li><code>yolo-tiny.cfg</code>&nbsp;is much smaller and based on the&nbsp;<a href="http://pjreddie.com/darknet/imagenet/#reference">Darknet reference network</a>. It processes images at 155 fps, here are weight files for&nbsp;<code>yolo-tiny.cfg</code>&nbsp;trained on&nbsp;<a href="http://pjreddie.com/media/files/yolo-tiny.weights">2007 train/val+ 2012 train/val</a>.</li>
</ul>
<h1>Model Architecture</h1>
<hr />
<p>I have used a pretrained Tiny Yolo architecture using&nbsp;<a href="https://pjreddie.com/darknet/yolo/">weights from here</a>&nbsp;and used for real time video detection</p>
<p><img src="https://cdn-images-1.medium.com/max/800/1*ZbmrsQJW-Lp72C5KoTnzUg.jpeg" alt="" width="800" height="384" /></p>
<h1>Data</h1>
<hr />
<p>This is a video dataset of real time traffic.&nbsp;</p>
<p>The task is to make bounding boxes around the moving vehicles and objects.</p>
<h1>Dependencies :</h1>
<hr />
<ul>
<li>Numpy</li>
<li>Pandas</li>
<li>Matplotlib</li>
<li>Sckit-Learn</li>
<li>Keras (Tensorflow Backend)</li>
</ul>
<h1><strong>Reference:</strong></h1>
<hr />
<p><a href="https://pjreddie.com/darknet/yolo/">https://pjreddie.com/darknet/yolo/</a></p>
<pre><code>./darknet yolo test cfg/yolo-small.cfg yolo-small.weights</code></pre>
