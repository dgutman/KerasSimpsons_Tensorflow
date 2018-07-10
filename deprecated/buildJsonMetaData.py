import json
import os
import sys
import cv2



imgSubjMetaData = {}

subjRootDir = "rawImageData/"

for dirName, subdirList, fileList in os.walk(subjRootDir):
    print('Found directory: %s' % dirName)
    imgSubjMetaData[dirName] = []

    for fname in fileList:
    	## Get the image dimensions as well
		imagePath = dirName + "/" +fname
		try:
			# image = cv2.imread(imagePath)
			# (h, w) = image.shape[:2]
			(h ,w ) = (256,256)
			fileMetaData = { "label": dirName.replace(subjRootDir,""), 'filename': fname, 'fileWpath': imagePath, 'imgDimensions': (h,w) }
			imgSubjMetaData[dirName].append(fileMetaData)
		except:
			print "Having some issue with",imagePath

print imgSubjMetaData
## This is currently a dictionary, need it to be flattened into a list of dictionaries so it's
## more webix friendly


simpsonsImgMetaData = []

for k,v in imgSubjMetaData.iteritems():
	## Going to generate a "row" for each Class
	if len(v) > 0:
		classData = { "label": k.replace(subjRootDir,""), "imgsInClass": len(v), "imgSet": v, "sampleImg": v[0]}
		simpsonsImgMetaData.append(classData)

with open("web/simpsonsImgMetaData.json","w") as fp:
	json.dump(simpsonsImgMetaData,fp)