# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
try:
    # for Python2
    from Tkinter import *
except ImportError:
    # for Python3
    from tkinter import *
from PIL import ImageTk
from helpers import cv2DrawText
from helpers import *


####################################
# Parameters
####################################
#change it to your images directory. Run this script separately for each folder
imgDir = "../../DataSets/HotailorPOC2/testImages"


#change it to your classes names
classes = ["__background__","curtain", "pillow", "bed", "lamp", "toilet", "sink", "tap", "towel"]


#no need to change these
drawingImgSize = 1000
boxWidth = 10
boxHeight = 2


####################################
# Main
####################################
# define callback function for tk button
def buttonPressedCallback(s):
    global global_lastButtonPressed
    global_lastButtonPressed = s

# create UI
objectNames = ["SHOW NEXT"]
tk = Tk()
w = Canvas(tk, width=len(objectNames) * boxWidth, height=len(objectNames) * boxHeight, bd = boxWidth, bg = 'white')
w.grid(row = len(objectNames), column = 0, columnspan = 2)
for objectIndex,objectName in enumerate(objectNames):
    b = Button(width=boxWidth, height=boxHeight, text=objectName, command=lambda s = objectName: buttonPressedCallback(s))
    b.grid(row = objectIndex, column = 0)

# loop over all images
imgFilenames = getFilesInDirectory(imgDir, ".jpg")
imgFilenames += getFilesInDirectory(imgDir, ".png")
for imgIndex, imgFilename in enumerate(imgFilenames):
    print (imgIndex, imgFilename)
    ##if os.path.exists(labelsPath):
      ##  print ("Skipping image {:3} ({}) since annotation file already exists: {}".format(imgIndex, imgFilename, labelsPath))
        ##continue

    # load image, ground truth rectangles and labels
    img = imread(os.path.join(imgDir,imgFilename))
    rectsPath = os.path.join(imgDir, imgFilename[:-4] + ".bboxes.tsv")
    labelsPath = os.path.join(imgDir, imgFilename[:-4] + ".bboxes.labels.tsv")
    rects = [ToIntegers(rect) for rect in readTable(rectsPath)]
    with open(labelsPath) as file:
        labels = [line.strip() for line in file]
    
          
    imgCopy = img.copy()

    # annotate each rectangle in turn
    for index in range(len(rects)):  

        label = labels[index]
        rect = rects[index]        
        classIndex = (classes.index(label))

        if classIndex == 0:
            color = (255, 0, 0)
        else:
            color = tuple(getColorsPalette()[classIndex])
        
        drawRectangles(imgCopy, [rect], color = color, thickness = 2)

        font = ImageFont.truetype(available_font, 18)
        text = classes[classIndex]
        textWidth = len(text)*13
        drawRectangles(imgCopy, [[rect[0],rect[1]-23,rect[0]+textWidth,rect[1]]], color = color, thickness = -1)
        cv2DrawText(imgCopy, (rect[0]+3,rect[1]-7), text, color = (255,255,255), colorBackground=color)

        # draw image in tk window
        imgTk, _ = imresizeMaxDim(imgCopy, drawingImgSize, boUpscale = True)
        imgTk = ImageTk.PhotoImage(imconvertCv2Pil(imgTk))
        label = Label(tk, image=imgTk)
        label.grid(row=0, column=1, rowspan=drawingImgSize)


        # busy-wait until button pressed
        ##global_lastButtonPressed = None
        ##while not global_lastButtonPressed:
        ##    tk.update_idletasks()
        ##    tk.update()

        # store result
        ##print ("Button pressed = ", global_lastButtonPressed)
        ##labels.append(global_lastButtonPressed)
    tk.update_idletasks()
    tk.update()
    ##writeFile(labelsPath, labels)
    # busy-wait until button pressed
    global_lastButtonPressed = None
    while not global_lastButtonPressed:
        tk.update_idletasks()
        tk.update()


    # store result
    print ("Button pressed = ", global_lastButtonPressed)
tk.destroy()
print ("DONE.")