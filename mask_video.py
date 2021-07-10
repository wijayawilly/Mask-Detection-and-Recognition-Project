import numpy as np
import imutils
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import winsound

# untuk memaksimalkan penggunaan GPU
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def detect_and_predict_mask(frame, faceNet, maskNet):
	#Mengambil height dengan width dari frame yang dibuat 
	#dan melakukan preprocessing menggunakan blobFromImage
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (150, 150),
		(104.0, 177.0, 123.0))

	#Menggunakan value dari blob untuk mendapatkan facedetection
	faceNet.setInput(blob)
	detections = faceNet.forward()

	#Mempersiapkan list yang akan digunakan
	faces = []
	locs = []
	preds = []
	pred_binary = []

	for i in range(0, detections.shape[2]):

		confidence = detections[0, 0, i, 2] # Merupakan probabilitas dari deteksi

		if confidence > 0.5: #Membuat treshold untuk minimal probabilitas harus lebih besar dari 0.5
			
			#Untuk mendapatkan koordinat awal dan akhir dari x dan y
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			#Untuk memastikan box dimana wajah terdeteksi tidak melebihi frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			#Mangambil wajah di dalam box, melakukan prediksi terhadap wajah tersebut, dan melakukan data preprocessing
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (150, 150))
			face = img_to_array(face)
			face = face / 255.0

			#Memasukkan kedalam list face dan locs
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	if len(faces) > 0:
		#Melakukan prediksi dan merubah format prediksi menjadi 0 atau 1
		treshhold = 0.5
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
		pred_binary = (preds>treshhold).astype(int)

	return (locs, preds, pred_binary)

#Face detection module dari caffe deep learning framework
prototxtPath = r"face_detector\deploy.prototxt" #deploy model
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel" #weights
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath) #face detector

#Load Model
maskNet = load_model("model_checkpoint\\weights-improvement-84.h5")

#Mengakses camera
vs = VideoStream(src=0).start()

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=600)
	(locs, preds, pred_binary) = detect_and_predict_mask(frame, faceNet, maskNet)
		
	#Melakukan looping untuk menampilkan deteksi	
	for (box, preds_binary, pred) in zip(locs, pred_binary, preds):
		(startX, startY, endX, endY) = box
		result = preds_binary
		
		label = (result == 1 and "Mask") or (result == 0 and "Mask")
		color = (0, 255, 0) if result == 1 else (0, 0, 255)
		if result == 0:
			freq = 5000
			duration = 35
			winsound.Beep(freq, duration)
		
		label = "{}: {:.2f}%".format(label, pred[0] * 100)
		cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_DUPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
	
	cv2.putText(frame,'Frame Pause : Press p', (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,255,0), 2)
	cv2.putText(frame,'Quit App : Press q', (10,45), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,255,0), 2)
	cv2.putText(frame,'Frame Continue : Press p while frame stops', (10,70), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,255,0), 2)
        
	cv2.imshow("Face Mask Detector", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("p"): #Pause
		freq = 1000
		duration = 35
		winsound.Beep(freq, duration)
		cv2.waitKey(0)

	if key == ord("q"): #Quit
		break

cv2.destroyAllWindows()
vs.stop()
