
from __future__ import print_function
from builtins import str
import  os
import numpy as np
import copy
import cv2
from PIL import Image, ImageFont, ImageDraw
from PIL.ExifTags import TAGS

available_font = "arial.ttf"
try:
    dummy = ImageFont.truetype(available_font, 16)
except:
    available_font = "FreeMono.ttf"


def imresizeMaxDim(img, maxDim, boUpscale = False, interpolation = cv2.INTER_LINEAR):
    scale = 1.0 * maxDim / max(img.shape[:2])
    if scale < 1  or boUpscale:
        img = imresize(img, scale, interpolation)
    else:
        scale = 1.0
    return img, scale

def imresize(img, scale, interpolation = cv2.INTER_LINEAR):
    return cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=interpolation)

def imread(imgPath, boThrowErrorIfExifRotationTagSet = True):
    if not os.path.exists(imgPath):
        print("ERROR: image path does not exist.")
        error

    rotation = rotationFromExifTag(imgPath)
    if boThrowErrorIfExifRotationTagSet and rotation != 0:
        print ("Error: exif roation tag set, image needs to be rotated by %d degrees." % rotation)
    img = cv2.imread(imgPath)
    if img is None:
        print ("ERROR: cannot load image " + imgPath)
        error
    if rotation != 0:
        img = imrotate(img, -90).copy()  # got this error occassionally without copy "TypeError: Layout of the output array img is incompatible with cv::Mat"
    return img

def rotationFromExifTag(imgPath):
    TAGSinverted = {v: k for k, v in TAGS.items()}
    orientationExifId = TAGSinverted['Orientation']
    try:
        imageExifTags = Image.open(imgPath)._getexif()
    except:
        imageExifTags = None

    # rotate the image if orientation exif tag is present
    rotation = 0
    if imageExifTags != None and orientationExifId != None and orientationExifId in imageExifTags:
        orientation = imageExifTags[orientationExifId]
        # print ("orientation = " + str(imageExifTags[orientationExifId]))
        if orientation == 1 or orientation == 0:
            rotation = 0 # no need to do anything
        elif orientation == 6:
            rotation = -90
        elif orientation == 8:
            rotation = 90
        else:
            print ("ERROR: orientation = " + str(orientation) + " not_supported!")
            error
    return rotation

def drawRectangles(img, rects, color = (0, 255, 0), thickness = 2):    
    for rect in rects:
        pt1 = tuple(ToIntegers(rect[0:2]))
        pt2 = tuple(ToIntegers(rect[2:]))
        cv2.rectangle(img, pt1, pt2, color, thickness)

def getColorsPalette():
    colors = [[255,0,0], [0,255,0], [0,0,255], [255,255,0], [255,0,255]]
    for i in range(5):
        for dim in range(0,3):
            for s in (0.25, 0.5, 0.75):
                if colors[i][dim] != 0:
                    newColor = copy.deepcopy(colors[i])
                    newColor[dim] = int(round(newColor[dim] * s))
                    colors.append(newColor)
    return colors

def ToIntegers(list1D):
    return [int(float(x)) for x in list1D]

def drawCrossbar(img, pt):
    (x,y) = pt
    cv2.rectangle(img, (0, y), (x, y), (255, 255, 0), 1)
    cv2.rectangle(img, (x, 0), (x, y), (255, 255, 0), 1)
    cv2.rectangle(img, (img.shape[1],y), (x, y), (255, 255, 0), 1)
    cv2.rectangle(img, (x, img.shape[0]), (x, y), (255, 255, 0), 1)

def imconvertPil2Cv(pilImg):
    rgb = pilImg.convert('RGB')
    return np.array(rgb).copy()[:, :, ::-1]

def imconvertCv2Pil(img):
    cv2_im = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_im)

def cv2DrawText(img, pt, text, color = (255,255,255), colorBackground = None):
    # Write some Text
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 0.6
    lineType               =2
    cv2.putText(img,text, 
        pt, 
        font, 
        fontScale,
        color,
        lineType)


def pilDrawText(pilImg, pt, text, textWidth=None, color = (255,255,255), colorBackground = None, font = ImageFont.truetype("arial.ttf", 16)):
    textY = pt[1]
    draw = ImageDraw.Draw(pilImg)
    if textWidth == None:
        lines = [text]
    else:
        lines = textwrap.wrap(text, width=textWidth)
    for line in lines:
        width, height = font.getsize(line)
        if colorBackground != None:
            draw.rectangle((pt[0], pt[1], pt[0] + width, pt[1] + height), fill=tuple(colorBackground[::-1]))
        draw.text(pt, line, fill = tuple(color), font = font)
        textY += height
    return pilImg

def drawText(img, pt, text, textWidth=None, color = (255,255,255), colorBackground = None, font = ImageFont.truetype("arial.ttf", 16)):
    pilImg = imconvertCv2Pil(img)
    pilImg = pilDrawText(pilImg,  pt, text, textWidth, color, colorBackground, font)
    return imconvertPil2Cv(pilImg)

def imWidth(input):
    return imWidthHeight(input)[0]

def imHeight(input):
    return imWidthHeight(input)[1]

def imWidthHeight(input):
    width, height = Image.open(input).size  # this does not load the full image
    return width, height

def imArrayWidth(input):
    return imArrayWidthHeight(input)[0]

def imArrayHeight(input):
    return imArrayWidthHeight(input)[1]

def imArrayWidthHeight(input):
    width = input.shape[1]
    height = input.shape[0]
    return width, height

def ptClip(pt, maxWidth, maxHeight):
    pt = list(pt)
    pt[0] = max(pt[0], 0)
    pt[1] = max(pt[1], 0)
    pt[0] = min(pt[0], maxWidth)
    pt[1] = min(pt[1], maxHeight)
    return pt

def deleteFile(filePath):
    if os.path.exists(filePath):
        os.remove(filePath)

def writeFile(outputFile, lines):
    with open(outputFile,'w') as f:
        for line in lines:
            f.write("%s\n" % line)

def writeTable(outputFile, table):
    lines = tableToList1D(table)
    writeFile(outputFile, lines)

def deleteFile(filePath):
    if os.path.exists(filePath):
        os.remove(filePath)

def tableToList1D(table, delimiter='\t'):
    return [delimiter.join([str(s) for s in row]) for row in table]

def getFilesInDirectory(directory, postfix = ""):
    fileNames = [s for s in os.listdir(directory) if not os.path.isdir(os.path.join(directory, s))]
    if not postfix or postfix == "":
        return fileNames
    else:
        return [s for s in fileNames if s.lower().endswith(postfix)]

def readTable(inputFile, delimiter='\t', columnsToKeep=None):
    lines = readFile(inputFile);
    if columnsToKeep != None:
        header = lines[0].split(delimiter)
        columnsToKeepIndices = listFindItems(header, columnsToKeep)
    else:
        columnsToKeepIndices = None;
    return splitStrings(lines, delimiter, columnsToKeepIndices)

def readFile(inputFile):
    #reading as binary, to avoid problems with end-of-text characters
    #note that readlines() does not remove the line ending characters
    with open(inputFile,'rb') as f:
        lines = f.readlines()
    return [removeLineEndCharacters(s) for s in lines]

def removeLineEndCharacters(line):
    if line.endswith(b'\r\n'):
        return line[:-2]
    elif line.endswith(b'\n'):
        return line[:-1]
    else:
        return line

def splitStrings(strings, delimiter, columnsToKeepIndices=None):
    table = [splitString(string, delimiter, columnsToKeepIndices) for string in strings]
    return table;

def splitString(string, delimiter='\t', columnsToKeepIndices=None):
    if string == None:
        return None
    items = string.decode('utf-8').split(delimiter)
    if columnsToKeepIndices != None:
        items = getColumns([items], columnsToKeepIndices)
        items = items[0]
    return items;



class Bbox:
    MAX_VALID_DIM = 100000
    left = top = right = bottom = None

    def __init__(self, left, top, right, bottom):
        self.left   = int(round(float(left)))
        self.top    = int(round(float(top)))
        self.right  = int(round(float(right)))
        self.bottom = int(round(float(bottom)))
        self.standardize()

    def __str__(self):
        return ("Bbox object: left = {0}, top = {1}, right = {2}, bottom = {3}".format(self.left, self.top, self.right, self.bottom))

    def __repr__(self):
        return str(self)

    def rect(self):
        return [self.left, self.top, self.right, self.bottom]

    def max(self):
        return max([self.left, self.top, self.right, self.bottom])

    def min(self):
        return min([self.left, self.top, self.right, self.bottom])

    def width(self):
        width  = self.right - self.left + 1
        assert(width>=0)
        return width

    def height(self):
        height = self.bottom - self.top + 1
        assert(height>=0)
        return height

    def surfaceArea(self):
        return self.width() * self.height()

    def getOverlapBbox(self, bbox):
        left1, top1, right1, bottom1 = self.rect()
        left2, top2, right2, bottom2 = bbox.rect()
        overlapLeft = max(left1, left2)
        overlapTop = max(top1, top2)
        overlapRight = min(right1, right2)
        overlapBottom = min(bottom1, bottom2)
        if (overlapLeft>overlapRight) or (overlapTop>overlapBottom):
            return None
        else:
            return Bbox(overlapLeft, overlapTop, overlapRight, overlapBottom)

    def standardize(self): #NOTE: every setter method should call standardize
        leftNew   = min(self.left, self.right)
        topNew    = min(self.top, self.bottom)
        rightNew  = max(self.left, self.right)
        bottomNew = max(self.top, self.bottom)
        self.left = leftNew
        self.top = topNew
        self.right = rightNew
        self.bottom = bottomNew

    def crop(self, maxWidth, maxHeight):
        leftNew   = min(max(self.left,   0), maxWidth)
        topNew    = min(max(self.top,    0), maxHeight)
        rightNew  = min(max(self.right,  0), maxWidth)
        bottomNew = min(max(self.bottom, 0), maxHeight)
        return Bbox(leftNew, topNew, rightNew, bottomNew)

    def isValid(self):
        if self.left>=self.right or self.top>=self.bottom:
            return False
        if min(self.rect()) < -self.MAX_VALID_DIM or max(self.rect()) > self.MAX_VALID_DIM:
            return False
        return True
