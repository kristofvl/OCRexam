import pdfplumber
import numpy as np
from PIL import Image, ImageDraw
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from skimage.transform import resize
import joblib

save_jpgs = False
out_IDs = True

def crop_(arr):
    """
    Crop a 2D array to the smallest bounding box containing all non-zero elements,
    while preserving the original aspect ratio by expanding the bounding box.
    """
    non_zeros = np.nonzero(arr)
    if len(non_zeros[0]) == 0:
        return np.array([])  # All zeros
    row_min, row_max = non_zeros[0].min(), non_zeros[0].max()
    col_min, col_max = non_zeros[1].min(), non_zeros[1].max()
    return arr[row_min:row_max + 1, col_min:col_max + 1]

def centr_(arr):
    # Create a larger array of zeros
    large_array = np.zeros((88,88), dtype=arr.dtype)
    # Calculate the offsets to center the small array
    height_offset = (88 - arr.shape[0]) // 2
    width_offset = (88 - arr.shape[1]) // 2
    # Place the small array in the center
    large_array[
        height_offset:height_offset + arr.shape[0],
        width_offset:width_offset + arr.shape[1]
    ] = arr
    return large_array

char_map = {0: ' ', 1: '█'}

# Load MNIST data and train RF to detect digits:
if 0:
     X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
     X = (X > 0.5).astype(int)  # Threshold at 0.5 (MNIST pixels are 0-255, normalized to 0-1)
     clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, max_depth=15,)
     for s in range(5):
          sample_index = s  # Change this to the index of the sample you want
          sample = X[sample_index].reshape(28, 28)
          ascii_pic = '\n'.join([''.join([char_map[pixel] for pixel in row]) for row in sample])
          print(ascii_pic)
          print(y[sample_index])
     clf.fit(X, y)
     joblib.dump(clf, 'mnist_rf_model.joblib')  ## save model

# Load model
loaded_clf = joblib.load('mnist_rf_model.joblib')

with pdfplumber.open("scans.pdf") as pdf:
    for i in range(0,len(pdf.pages),1):
        img = pdf.pages[i].to_image(resolution=300).original
        # stage 0: crop out large top-right chunk:
        c_img = img.crop( (img.width-820, 370, img.width-90, 740) ).convert("L") 

        # stage 1: search the first large vertical empty space:
        img = ( np.array(c_img) < 64 ).astype(np.uint8) # pixels are between 0 - 255, threshold 1/4
        vsums = np.sum(img, axis=1)
        win_size = 9 # search for 9 subsequent rows where the sum is below 3:
        convolution = np.convolve(vsums < 3, np.ones(win_size, dtype=int), mode='valid')
        start = np.argwhere(convolution == win_size)
        start_up = start[0][0]+win_size
        start_dn = start[0][0]+110;
        c_img = c_img.crop( (0, start_up, c_img.width, start_dn) )

        # stage 2: search for the best matching 500 x 7 black line:
        img = ( np.array(c_img) < 64 ).astype(np.uint8) # pixels are between 0 - 255, threshold 1/4
        max_x = max_y = tmax = 0; wl = 500; wh = 7
        for x in range(len(img[0])-wl):
            for y in range(len(img)-wh):
                if np.sum(img[y:y+wh,x:x+wl]) > tmax:
                    tmax = np.sum(img[y:y+wh,x:x+wl])
                    max_x = x; max_y = y
        start_up = np.max( [max_y - 70, 0] )
        start_dn = max_y+1
        start_left = np.max([max_x - 10,0])
        start_right = np.min([max_x + wl + 1,len(img[0])])
        c_img = c_img.crop( (start_left, start_up, start_right, start_dn) )

        # stage 3: search from right for the digit markers and extract digits:
        img = img[start_up:start_dn,start_left:start_right]
        vsums = np.sum(img, axis=0)
        x = len(vsums)-2;
        draw = ImageDraw.Draw(c_img)
        markers = []
        while x > 0:
            if vsums[x]>19 and vsums[x+1]>19:
                markers.append(x)
                if x>50-1: x = x - 50  # jump decent amount to left
                else: x = 0
            x = x-1
        if len(markers) > 1:
            for m in range(1, len(markers)):
                draw.rectangle( (markers[m]+2,1, markers[m-1]-2,start_dn-start_up-1) , outline="red", width=1)
        if len(markers) <= 7:
            continue

        # stage 4: identify single digits
        id = ""
        for m in range(1, 8):
            sample = (np.array(c_img) < 200).astype(np.uint8)
            sample = crop_(sample[2:-3,markers[m]+3:markers[m-1]-3])  ## carve out the digit
            if len(sample)<60:
                _height, _width = sample.shape
                scale_factor = 70 / _height
                target_width = np.min( [int(_width * scale_factor), 70] )
                sample = resize(sample, (70, target_width), mode='constant', preserve_range=True)
            sample = centr_(sample)
            sample = resize(sample, (28, 28), mode='constant', preserve_range=True)
            sample = (sample > 0.005).astype(int)
            if m==1: samples = sample
            else: samples = np.hstack((sample, samples))
            sample = sample.flatten().reshape(1, -1)
            pred = loaded_clf.predict(sample)
            confidence_scores = loaded_clf.predict_proba(sample)
            output = ", ".join([f"{score:.3f}" for score in confidence_scores[0]])
            print([ pred[0], output])
            id = pred[0]+id
        print("Predicted:", id)
        if out_IDs:
            ascii_pic = '\n'.join([''.join([char_map[pixel] for pixel in row]) for row in samples])
            print(ascii_pic)

        if save_jpgs:
            c_img = c_img.crop( (markers[7], 0, markers[0], c_img.height) )
            c_img.save("out"+str(i)+"_"+id+".jpg")  ## for debugging purposes

