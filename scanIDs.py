import pdfplumber
import numpy as np
from PIL import Image, ImageDraw
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from skimage.transform import resize
import joblib, os

class ScanID:

	def __init__(self, model='mnist_rf_model.joblib', save_jpgs=False, out_IDs=True):
		self.out_IDs = out_IDs; self.save_jpgs = save_jpgs
		self.char_map = {0: ' ', 1: '█'}
		if not os.path.isfile(model):  # Load MNIST data and train RF to detect digits
			X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
			X = (X > 0.5).astype(int)  # Threshold at 0.5 (MNIST pixels are 0-255, normalized to 0-1)
			self.clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=9,)
			for s in range(5):
				self.ascii_pic(X[s].reshape(28, 28)); print(y[s])
			self.clf.fit(X, y)
			joblib.dump(self.clf, model)  # save trained odel
		else:
			self.clf = joblib.load(model)  # load trained RF model

	def ascii_pic(self, arr):  # display 2d binary array as an ascii picture
		print('\n'.join([''.join([self.char_map[pixel] for pixel in row]) for row in arr]))

	def crop_(self, arr):  # Crop 2d array to smallest bounding box containing all non-zero elements
		non_zeros = np.nonzero(arr)
		if len(non_zeros[0]) == 0:
			return np.array([])  # All zeros
		row_min, row_max = non_zeros[0].min(), non_zeros[0].max()
		col_min, col_max = non_zeros[1].min(), non_zeros[1].max()
		return arr[row_min:row_max + 1, col_min:col_max + 1]

	def centr_(self, arr):
		ret_array = np.zeros((88,88), dtype=arr.dtype)  # Create a larger array of zeros
		height_offset = (88 - arr.shape[0]) // 2  # Calculate the offsets to center the small array
		width_offset = (88 - arr.shape[1]) // 2
		ret_array[ height_offset:height_offset + arr.shape[0],
		    		   width_offset:width_offset + arr.shape[1]
		] = arr  # Place the small array in the center
		return ret_array

	def match_vspace_(self, img, win_size=9, sum_th=3):  # search 9 subsequent rows where sum < 3
		img = ( np.array(img) < 64 ).astype(np.uint8) # pixels are between 0 - 255, threshold 1/4
		vsums = np.sum(img, axis=1)
		convolution = np.convolve(vsums < sum_th, np.ones(win_size, dtype=int), mode='valid')
		start = np.argwhere(convolution == win_size)
		return [start[0][0]+win_size, start[0][0]+110, img]

	def match_line_(self, img, wl = 500, wh = 7):  # search best matching horizontal line wl x wh
		img = ( np.array(img) < 64 ).astype(np.uint8)  # pixels are between 0 - 255, threshold 1/4
		max_x = max_y = tmax = 0
		for x in range(len(img[0])-wl):
			for y in range(len(img)-wh):
				if np.sum(img[y:y+wh,x:x+wl]) > tmax:
					tmax = np.sum(img[y:y+wh,x:x+wl]); max_x = x; max_y = y
		return [ np.max( [max_y - 70, 0] ), max_y+1,
			np.max([max_x - 10,0]), np.min([max_x + wl + 1,len(img[0])]), img ]

	def match_dspaces_(self, img, dw=50, vs=19):  # search digit spaces of minimally 50 pixels wide
		vsums = np.sum(img, axis=0)
		markers = []
		x = len(vsums)-2;
		while x > 0:
			if vsums[x]>vs and vsums[x+1]>vs:
				markers.append(x)
				if x>dw-1: x = x - dw  # jump decent amount of pixels to the left
				else: x = 0
			x = x-1
		return markers

	def run(self, pdf_file):
		with pdfplumber.open(pdf_file) as pdf:
			for i in range(0,len(pdf.pages),1):
				img = pdf.pages[i].to_image(resolution=300).original
				# stage 0: crop out large top-right chunk:
				c_img = img.crop( (img.width-820, 370, img.width-90, 740) ).convert("L")
				# stage 1: search the first large vertical empty space:
				[start_up, start_dn, img] = self.match_vspace_(c_img)
				c_img = c_img.crop( (0, start_up, c_img.width, start_dn) )
				# stage 2: search for the best matching 500 x 7 black line:
				[start_up, start_dn, start_left, start_right, img] = self.match_line_(c_img)
				c_img = c_img.crop( (start_left, start_up, start_right, start_dn) )
				# stage 3: search from right for the digit markers and extract digits:
				img = img[start_up:start_dn,start_left:start_right]
				markers = self.match_dspaces_(img)
				if len(markers) > 1:
					for m in range(1, len(markers)):  # draw markers over potential digits in image
						draw = ImageDraw.Draw(c_img)
						draw.rectangle( (markers[m]+2,1, markers[m-1]-2, start_dn-start_up-1),
							outline="red", width=1)
				if len(markers) <= 7: continue
				# stage 4: identify single digits
				id = ""  # to build up the ID
				samples = []  # collect binary bitmap samples to visualize here
				for m in range(1, 8):
					sample = (np.array(c_img) < 200).astype(np.uint8)
					sample = self.crop_(sample[2:-3,markers[m]+3:markers[m-1]-3])  # carve out digit
					if len(sample)<60:  # zoom in on bitmap if it is too small:
						_height, _width = sample.shape
						scale_factor = 70 / _height
						_width = np.min( [int(_width * scale_factor), 70] )
						sample = resize(sample, (70, _width), mode='constant', preserve_range=True)
					sample = self.centr_(sample)
					sample = resize(sample, (28, 28), mode='constant', preserve_range=True)
					sample = (sample > 0.005).astype(int)
					if m==1: samples = sample
					else: samples = np.hstack((sample, samples))
					sample = sample.flatten().reshape(1, -1)
					pred = self.clf.predict(sample)
					confidence_scores = self.clf.predict_proba(sample)
					output = ", ".join([f"{score:.3f}" for score in confidence_scores[0]])
					print([ pred[0], output])
					id = pred[0]+id
				print("Predicted:", id)
				if self.out_IDs:
					self.ascii_pic(samples)
				if self.save_jpgs:
					c_img = c_img.crop( (markers[7], 0, markers[0], c_img.height) )
					c_img.save("out"+str(i)+"_"+id+".jpg")  ## for debugging purposes

# scan the pages of the PDF document for IDs:
ScanID('mnist_rf.joblib').run('scans.pdf')
