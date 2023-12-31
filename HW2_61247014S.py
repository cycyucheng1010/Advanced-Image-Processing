import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtWidgets import QLabel,QPushButton,QFileDialog
from PyQt5.QtGui import QIcon,QImage,QPixmap
from PyQt5.QtCore import pyqtSlot


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
		self.label1.setGeometry(25,75,300,200)
		self.label2.setGeometry(375,75,300,200)
		# button
		self.btn1 = QPushButton('Open File',self)
		self.btn1.move(325,150)
		self.btn1.clicked.connect(self.file_manager)
		self.btn2 = QPushButton('Rotate Image',self)
		self.btn2.move(480,30)
		self.btn2.clicked.connect(self.rotate_img)
		self.btn2.hide()
		self.btn3 = QPushButton('Histogram',self)
		self.btn3.move(300,30)
		self.btn3.clicked.connect(self.show_histogram)
		self.btn3.hide()
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
		self.btn1.move(125,30)
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





if __name__ == '__main__':
	app = QApplication(sys.argv)
	ex = MyWidget()
	sys.exit(app.exec_())