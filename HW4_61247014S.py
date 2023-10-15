import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtWidgets import QLabel,QPushButton,QFileDialog,QInputDialog,QGridLayout,QLineEdit
from PyQt5.QtGui import QIcon,QImage,QPixmap
from PyQt5.QtCore import pyqtSignal
from skimage.util import random_noise



class NumberInputWindow(QWidget):
	dataSaved = pyqtSignal(list)
	def __init__(self):
		super().__init__()
		self.initUI()
		self.setWindowIcon(QIcon('NTNU.ico'))
		self.setWindowTitle('number input window')
		self.setGeometry(100,100,400,300)
		self.data = np.array([])

	def initUI(self):
		layout = QGridLayout()
		self.input_field = []
		for row in range(5):
			for col in range(5):
				input_field = QLineEdit()
				self.input_field.append(input_field)
				layout.addWidget(input_field,row,col)
		self.save_button=QPushButton('save',self)
		self.save_button.clicked.connect(self.save_data)
		layout.addWidget(self.save_button,5,0,1,5)
		self.setLayout(layout)
	def save_data(self):
		self.data = [[0 for _ in range(5)] for _ in range(5)]

		for row in range(5):
			for col in range(5):
				index = row * 5 + col
				text = self.input_field[index].text()
				if text.strip():
					try:
						value = float(text)
						self.data[row][col] = value
					except ValueError:
						pass
		print('data:',self.data)
		self.dataSaved.emit(self.data)
		self.close()
class MyWidget(QWidget):
	
	def __init__(self):
		super().__init__()
		self.initUI()
	
	def initUI(self):
		#title and icon
		self.setWindowTitle('AIS61247014S')
		self.setGeometry(500,500,700,300)
		self.setWindowIcon(QIcon('NTNU.ico'))
		#label

		self.label1 = QLabel(self)
		self.label2 = QLabel(self)
		self.label3 = QLabel(self)
		self.label4 = QLabel(self)
		self.label5 = QLabel(self)
		self.label6 = QLabel(self)
		self.label1.setGeometry(25,75,300,200)
		self.label2.setGeometry(375,75,300,200)
		self.label3.setGeometry(725,75,300,200)
		self.label4.setGeometry(25,350,300,200)
		self.label5.setGeometry(375,350,300,200)
		self.label6.setGeometry(725,350,300,200)
		# button
		self.btn1 = QPushButton('Open File',self)
		self.btn1.move(325,150)
		self.btn1.clicked.connect(self.file_manager)
		
		self.btn2 = QPushButton('Rotate Image',self)
		self.btn2.move(125,30)
		self.btn2.clicked.connect(self.rotate_img)
		self.btn2.hide()
		
		self.btn3 = QPushButton('Histogram',self)
		self.btn3.move(225,30)
		self.btn3.clicked.connect(self.show_histogram)
		self.btn3.hide()

		self.btn4 = QPushButton('gaussian noise',self)
		self.btn4.move(325,30)
		self.btn4.clicked.connect(self.input_gaussian_noise_parameter)
		self.btn4.hide()

		self.btn5 = QPushButton('salt and pepper noise',self)
		self.btn5.move(425,30)
		self.btn5.clicked.connect(self.input_salt_gaussian_parameter)
		self.btn5.hide()

		self.btn6 = QPushButton('convolution',self)
		self.btn6.move(560,30)
		self.btn6.clicked.connect(self.convolution)
		self.btn6.hide()
		# show the window
		self.show()

	def file_manager(self):
		try:
			filename =QFileDialog.getOpenFileName(self,'open image',filter='(*.ppm *.jpg *.bmp);;*.ppm ;;*.jpg ;;*.bmp') #os.getcwd(),
			print(filename)
			self.img = cv2.imread(filename[0])
			down_width = 300
			down_height = 200
			down_points = (down_width, down_height)
			self.img = cv2.resize(self.img, down_points, interpolation= cv2.INTER_LINEAR)
			self.show_image()
		except Exception as e:
			print(e)

	def show_image(self):
		self.height,self.width,self.channel = self.img.shape
		self.bytesPerline = self.channel * self.width
		qimg = QImage(self.img,self.width,self.height,self.bytesPerline,QImage.Format_RGB888)
		self.canvas1 = QPixmap(360,360).fromImage(qimg)
		self.canvas2 = self.canvas1
		self.label1.setPixmap(self.canvas1)
		self.label2.setPixmap(self.canvas2)
		self.btn2.show()
		self.btn3.show()
		self.btn4.show()
		self.btn5.show()
		self.btn6.show()
		self.btn1.move(25,30)
		self.canvas2.save('rotate.png','png')

	def rotate_img(self):
		try:
			self.canvas2 = cv2.rotate(cv2.imread('rotate.png'),cv2.ROTATE_180)
			self.height,self.width,self.channel = self.canvas2.shape
			qimg = QImage(self.canvas2,self.width,self.height,self.bytesPerline,QImage.Format_RGB888)
			self.canvas2 = QPixmap(360,360).fromImage(qimg)
			self.label2.setPixmap(self.canvas2)
			self.canvas2.save('rotate.png','png')
		except Exception as e:
			print(e)

	def save_histogram(self):
		self.gray_image = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
		plt.title('image')
		plt.xlabel('Intensity')
		plt.ylabel('Frequency')
		#hist,bin = np.histogram(self.gray_image.ravel(),256,[0,255])
		plt.xlim([0,255])
		#plt.plot(hist)
		plt.title('gray level histogram')
		plt.hist(self.gray_image.ravel(), 256, [0, 256])
		plt.savefig('gray_level_histogram.png')
		plt.close()

	def show_histogram(self):
		try:
			self.save_histogram()
			self.gray_histogram = cv2.imread('gray_level_histogram.png')
			down_width = 300
			down_height = 200
			down_points = (down_width, down_height)
			self.gray_histogram = cv2.resize(self.gray_histogram, down_points, interpolation= cv2.INTER_LINEAR)
			self.canvas2 = self.gray_histogram
			qimg = QImage(self.canvas2,self.width,self.height,self.bytesPerline,QImage.Format_RGB888)
			self.canvas2 = QPixmap(360,360).fromImage(qimg)
			self.label2.setPixmap(self.canvas2)
		except Exception as e:
			print(e)
	
	def input_salt_gaussian_parameter(self):
		try:
			percentage, p_ok = QInputDialog.getDouble(self,'input dialog','input percentage')
			if p_ok & (type(percentage)is int) or (type(percentage)is float):
				self.noise_percentage = percentage
				print(self.noise_percentage)
				self.salt_and_pepper_noise()
		except Exception as e:
			print(e)

	def input_gaussian_noise_parameter(self):
		try:
			standard_deviation,s_ok = QInputDialog.getDouble(self, 'input dialog', 'standard_deviation')
			if s_ok & ((type(standard_deviation) is int) or (type(standard_deviation) is float)):
					self.noise_parameter = standard_deviation
					print('noise std',self.noise_parameter)
					self.gaussian_noise()
			elif (type(standard_deviation) is not float) & s_ok:
					print('standard deviation type error')
		except Exception as e:
			print(e)

	def gaussian_noise(self):
		height = self.height
		width = self.width
		sigma = self.noise_parameter
		mean =25
		iterations =100

		# pure noise 
		pure_noise = np.full((height, width),255, dtype = np.uint8)
		for i in range(iterations):
			np.random.normal
			r1 = np.random.rand(height, width)
			r2 = np.random.rand(height, width)
			z1 = sigma * np.cos(2 * np.pi * r2) * np.sqrt(-2 * np.log(r1))
			z2 = sigma * np.sin(2 * np.pi * r2) * np.sqrt(-2 * np.log(r1))

			f_prime = pure_noise + z1
			f_prime_neighbor = np.roll(pure_noise, shift=(0, 1)) + z2

			pure_noise = np.clip(f_prime, 0, 255)
			noisy_image_neighbor = np.clip(f_prime_neighbor, 0,255)

			if np.all(np.abs(pure_noise - f_prime) < 1e-5) and np.all(np.abs(noisy_image_neighbor - f_prime_neighbor) < 1e-5):
				break

		cv2.imwrite('noise_pure.png',pure_noise)

		# mix noise
		noisy_image = self.img
		for i in range(iterations):

			r1 = np.random.rand(height, width)
			r2 = np.random.rand(height, width)
			z1 = sigma * np.cos(2 * np.pi * r2) * np.sqrt(-2 * np.log(r1))
			z2 = sigma * np.sin(2 * np.pi * r2) * np.sqrt(-2 * np.log(r1))
			z1 = z1[..., np.newaxis]  
			z2 = z2[..., np.newaxis]  
			
			f_prime = noisy_image + z1
			f_prime_neighbor = np.roll(noisy_image, shift=(0, 1)) + z2

			
			noisy_image = np.clip(f_prime, 0, 255)
			noisy_image_neighbor = np.clip(f_prime_neighbor, 0,255)

			
			if np.all(np.abs(noisy_image - f_prime) < 1e-5) and np.all(np.abs(noisy_image_neighbor - f_prime_neighbor) < 1e-5):
				break
		cv2.imwrite('noise_mix.png',noisy_image)
		self.show_noise()
	
	def salt_and_pepper_noise(self):
		#pure noise
		height = self.height
		width = self.width
		noise_percentage = self.noise_percentage
		pure_noise = np.full((height, width),255, dtype = np.uint8)
		pure_noise_image = random_noise(pure_noise,mode='s&p',amount = noise_percentage/100)
		pure_noise_image = (pure_noise_image *255).astype(np.uint8)
		cv2.imwrite('noise_pure.png',pure_noise_image)

		# mix noise
		mix_noise_image = self.img
		mix_noise_image = mix_noise_image.astype(float)/255.0
		self.noise_image = random_noise(mix_noise_image,mode='s&p',amount = self.noise_percentage/100)
		self.noise_image = (self.noise_image *255).astype(np.uint8)
		cv2.imwrite('noise_mix.png',self.noise_image)
		self.show_noise()
		

	def show_noise(self):
		noise_image_mix = cv2.imread('noise_mix.png')
		noise_image_pure = cv2.imread('noise_pure.png')
		self.setGeometry(500,500,1100,600)
		noise = QImage(noise_image_pure,self.width,self.height,self.bytesPerline,QImage.Format_RGB888)
		qimg = QImage(noise_image_mix,self.width,self.height,self.bytesPerline,QImage.Format_RGB888)
		self.canvas3 = QPixmap(360,360).fromImage(qimg)
		self.label3.setPixmap(self.canvas3)
		self.canvas2 = QPixmap(360,360).fromImage(noise)
		self.label2.setPixmap(self.canvas2)
		self.pure_noise = noise_image_pure
		self.mix_noise = noise_image_mix
		self.save_noise_histogram(self.img,'origial')
		self.save_noise_histogram(self.pure_noise,'pure')
		self.save_noise_histogram(self.mix_noise,'mix')

	def save_noise_histogram(self,image,name):
			gray_level_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
			plt.title('image')
			plt.xlabel('Intensity')
			plt.ylabel('Frequency')
			#plt.xlim([0,255])
			plt.title('gray level histogram')
			plt.hist(gray_level_image.ravel(), 256, [0, 256])
			plt.savefig(name+'_gray_level_histogram.png')
			plt.close()
			self.show_noise_histogram(self,name)

	def show_noise_histogram(self,image,name):
		self.gray_histogram = cv2.imread(name+'_gray_level_histogram.png')
		down_width = 300
		down_height = 200
		down_points = (down_width, down_height)
		self.gray_histogram = cv2.resize(self.gray_histogram, down_points, interpolation= cv2.INTER_LINEAR)
		if name =='origial':
			self.canvas4 = self.gray_histogram
			qimg = QImage(self.canvas4,self.width,self.height,self.bytesPerline,QImage.Format_RGB888)
			self.canvas4 = QPixmap(360,360).fromImage(qimg)
			self.label4.setPixmap(self.canvas4)
		elif name=='pure':
			self.canvas5 = self.gray_histogram
			qimg = QImage(self.canvas5,self.width,self.height,self.bytesPerline,QImage.Format_RGB888)
			self.canvas5 = QPixmap(360,360).fromImage(qimg)
			self.label5.setPixmap(self.canvas5)
		elif name=='mix':
			self.canvas6 = self.gray_histogram
			qimg = QImage(self.canvas6,self.width,self.height,self.bytesPerline,QImage.Format_RGB888)
			self.canvas6 = QPixmap(360,360).fromImage(qimg)
			self.label6.setPixmap(self.canvas6)
	
	def convolution(self):
		self.number_input_window = NumberInputWindow()
		self.core=self.number_input_window.dataSaved.connect(self.convolution_calculation)
		self.number_input_window.show()
	
	def convolution_calculation(self,data,padding=2, strides=1):
		self.gray_image = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
		#cv2.imwrite('test.png',self.gray_image)
		kernel = np.flipud(np.fliplr(data))
		print('core',kernel)
		xKernShape = kernel.shape[0]
		yKernShape = kernel.shape[1]
		xImgShape = self.gray_image.shape[0]
		yImgShape = self.gray_image.shape[1]
		xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
		yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
		output = np.zeros((xOutput, yOutput))

		if padding != 0:
			imagePadded = np.zeros((self.gray_image.shape[0] + padding*2, self.gray_image.shape[1] + padding*2))
			imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = self.gray_image
			print(imagePadded)
		else:
			imagePadded = self.gray_image
		
		for y in range(self.gray_image.shape[1]):
			if y > self.gray_image.shape[1] - yKernShape:
				break
			if y % strides == 0:
				for x in range(self.gray_image.shape[0]):
					if x > self.gray_image.shape[0] - xKernShape:
						break
					try:
						if x % strides == 0:
							output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
					except:
						break
		cv2.imwrite('2Dconvolved.png',output)
		self.conv2D = cv2.imread('2Dconvolved.png')
		self.canvas2 = self.conv2D
		qimg = QImage(self.canvas2,self.width,self.height,self.bytesPerline,QImage.Format_RGB888)
		self.canvas2 = QPixmap(360,360).fromImage(qimg)
		self.label2.setPixmap(self.canvas2)
		

if __name__ == '__main__':
	app = QApplication(sys.argv)
	ex = MyWidget()
	sys.exit(app.exec_())