# Table of contents
- [Project description](#project-description)
- [Results and learnings](#results-and-learnings)
    - [Initial assumptions](#initial-assumptions)
    - [Dataset](#dataset)
    - [Training and evaluation results](#training-and-evaluation-results)    
    - [Using the model](#using-the-model)
- [Run sample](#run-sample)
    - [Setup](#setup)
    - [Train and evaluate the model](#train-and-evaluate-the-model-using-hotailorpoc2-sample-dataset) 
- [Code highlights](#code-highlights)
- [Use custom dataset](#use-custom-dataset)

<br><br>

# Project description 
[[back to the top]](#table-of-contents)

This **POC** is using **CNTK 2.1** to train model for **multiclass classification of images**. Our model is able to recognize specific objects (i.e. toilet, tap, sink, bed, lamp, pillow) connected with picture types we are looking for. It plays a big role in a process which will be used to **classify pictures from different hotels and determine whether it's a picture of bathroom, bedroom, hotel front, swimming pool, bar, etc**. That final classification will be made based on objects that were detected in those pictures. 

What can you find inside:
- How to train a **multiclass classificator for images** using [**CNTK (Cognitive Toolkit)**](https://github.com/Microsoft/CNTK) and [**FasterRCNN**](https://github.com/Microsoft/CNTK/tree/master/Examples/Image/Detection/FasterRCNN)
- Training using [**Transfer Learning**](https://docs.microsoft.com/en-us/cognitive-toolkit/Build-your-own-image-classifier-using-Transfer-Learning) with pretrained [**AlexNet**](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) model
- How to **prepare and label images in a dataset** used for training and testing the model 
- Working example with all the data and pretrained models

If you would like to know how to use such model, you can check [**this project**](https://github.com/karolzak/CNTK-Python-Web-Service-on-Azure) to find out how to write a simple **RESTfull, Python-based web service and deploy it to Azure Web Apps with your own model**.

<br><br>

# **Results and learnings**
[[back to the top]](#table-of-contents)

***Disclaimer:***
*This POC and all the learnings you can find bellow is an outcome of close cooperation between Microsoft and [Hotailors](https://hotailors.com/). Our combined team spent total of 3 days to prepare and label data, finetune parameters and train the model.*

<br>

### Initial assumptions ###
[[back to the top]](#table-of-contents)

- Due to limited time and human resources we decided to create this POC for just 2 of almost 20 different types of pictures we would like to classify in final product
- Each type of picture (i.e. `bedroom, bathroom, bar, lobby, hotel front, restaurant`) can consists of different objects (i.e. `toilet, sink, tap, towell, bed, lamp, curtain, pillow`) which are strongly connected with that speciifc picture type. 

- For our POC we used 2 picture types with 4 objects/classes per each:

    bedroom     |  bathroom
    :----------:|:----------:
    pillow      | tap 
    bed         | sink
    curtain     | towel
    lamp        | toilet

- At this time we focused only on detecting those specific objects for each picture type. Outcomes of evaluation should later be analyzed either by some simple algorithm or another model to match an image with one of the picture types we are looking for 

<br><br>

### Dataset ###
[[back to the top]](#table-of-contents)
- We wanted to be as close as possible to real world scenarios so our dataset consists of **real pictures from different hotels** all over the world. Images where provided by Hotailors team
- In our POC we used images scalled to **max of 1000px on the wide side**
- **Every picture usually consists of multiple types of objects** we are looking for
- We used total of **113 images** to train and test our model from which we used:
    - **82 images in `positive` set** for training the model. We have about 50/50 split between `bathroom` and `bedroom` pictures

        Bathroom positive sample   |  Bedroom positive sample
        :-------------------------:|:-------------------------:
        ![](doc/positive_bathroom.jpg) | ![](doc/positive_bedroom.jpg)

    - **11 images in `negative` set** for training the model. Those images should not contain any objects that we are interested in detecting
        
        Negative sample 1  |  Negative sample 2
        :-------------------------:|:-------------------------:
        ![](doc/negative1.jpg) | ![](doc/negative2.jpg)

    - **20 images in `testImages` set** for testing and evaluating the model. We have about 50/50 split between `bathroom` and `bedroom` pictures
    
        Bathroom test sample   |  Bedroom test sample
        :-------------------------:|:-------------------------:
        ![](doc/test_bathroom.jpg) | ![](doc/test_bedroom.jpg)

- After we tagged all of the images from `HotailorPOC2` dataset we analyzed them to verify how many tagged objects per each class we have. It is suggested to use about 20-30% of all data in dataset as test data. Looking at our numbers below we did quite ok but there's still some room for improvement

    object/class name | # of tagged objects in positive/train set   | # of tagged objects in test set | % of tagged objects in relation to all objects
    :-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
    sink | 46 | 10 | 18
    pillow | 98 | 27 | 22
    toilet | 34 | 7 | 17
    lamp | 69 | 18 | 21
    curtain | 78 | 16 | 17
    towel | 30 | 14 | 32
    tap | 44 | 9 | 17
    bed | 53 | 12 | 18

<br><br>

### Training and evaluation results ###
[[back to the top]](#table-of-contents)

- After training and evaluating our model we achieved following results:

    ```
    Evaluating Faster R-CNN model for 20 images.
    Number of rois before non-maximum suppression: 550
    Number of rois  after non-maximum suppression: 87
    AP for            sink = 0.4429
    AP for          pillow = 0.1358
    AP for          toilet = 0.8095
    AP for            lamp = 0.5404
    AP for         curtain = 0.7183
    AP for           towel = 0.0000
    AP for             tap = 0.1111
    AP for             bed = 0.8333
    Mean AP = 0.4489
    ```
- As you can see above, some of the results are not too good. For example: `pillow` and `tap` average precision for test set is extremely low and for `towel` it even shows 0.0000 which may indicate some problems with our dataset or tagged objects. We will definitely need to look into it and check if we are able to somehow improve those results

- Even though the Mean Average Precision values are not perfect we still were able to get some decent results:

    ![](doc/good_bathroom.jpg)  
    ![](doc/good_bedroom.jpg)
    
- Some of the results include mistakes. But those clearly look like anomalies which should be fairly easy to catch in further classification of picture type

    *Picture below shows how our model classified single region (yellow) as **`bed`** object although it's clearly not there:*![](doc/bad_bathroom.jpg)   

    *Another picture shows how our model classified single region as **`towel`** object although it's clearly not there:*![](doc/bad_bedroom.jpg)

- Ofcourse sometimes there are some really ugly results which may be hard to use for further classification:

    ![](doc/ugly_bathroom.jpg)

     *Next picture shows our model wasn't able to find any objects. We need to verify if it's because of wrongly tagged data in HotailorPOC2 or is it some kind of issue with [selective search algorithm](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf) and it simply didn't find any [regions of interest](https://en.wikipedia.org/wiki/Region_of_interest) for evaluation* 
    ![](doc/ugly_bathroom2.jpg)

<br><br>

### Using the model ### 
[[back to the top]](#table-of-contents)

Final model will be used in form of web service running on Azure and that's why I prepared a sample RESTful web service written with Python using Flask module. This web service makes use of our trained model and provides API which takes images as an input for evaluation and returns either a cloud of tags or tagged images. Project also describes how to easily deploy this web service to Azure Web Apps with custom Python environment and required dependencies.

You can find running web service hosted on Azure Web Apps [here](http://cntkpywebapptest1.azurewebsites.net/), and project with code and deployement scripts can be found on [GitHub](https://github.com/karolzak/CNTK-Python-Web-Service-on-Azure).

![Demo](doc/iexplore_2017-09-26_23-09-42.jpg)

*Sample request and response in Postman:*
![Demo](doc/Postman_2017-09-26_22-50-06.jpg)

<br><br>


# Run sample

## Setup
[[back to the top]](#table-of-contents)


1. **Download content of this repo**

    You can either clone this repo or just download it and unzip to some folder

2. **Setup Python environment**

    In order for scripts to work you should have a proper Python environment. If you don't already have it setup then you should follow one of the online tutorials. To setup Python environment and all the dependencies required by CNTK on my local Windows machine, I used [scripted setup tutorial for Windows](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-windows-binary-script). If you're using Linux then you might want to look into one of these [tutorials](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-your-machine).
    Just bear in mind that this project was **developed and tested with CNTK 2.1 and it wasn't tested for any other version**.
    
    Even after setting up Python environment properly you might still witness some errors when running Python scripts. Most of those errors are related to missing modules or some 3rd party frameworks and tools (i.e. [GraphViz](http://www.graphviz.org/)). Missing modules can be easily [pip installed](https://packaging.python.org/tutorials/installing-packages/) and most of the required ones can be found in `requirements.txt` files for each folder with Python scripts.

    Please report if you'll find any errors or missing modules, thanks!

3. **Download hotel pictures dataset (HotailorPOC2) and pretrained AlexNet model used for Transfer Learning**

    Go to [Detection/FasterRCNN](Detection/FasterRCNN) folder in the location were you unzipped this repo and run `install_data_and_model.py`. It will automatically download the `HotailorPOC2` dataset, pretrained AlexNet model and will generate mapping files required to train the model.
    
## Train and evaluate the model using HotailorPOC2 sample dataset
[[back to the top]](#table-of-contents)

### Training the model ###

After you go through setup steps you can start training your model.

In order to do it you need to run `FasterRCNN.py`script in [Detection/FasterRCNN](Detection/FasterRCNN). 

I'm working on Windows 10 so I run the script from Anaconda Command Prompt which should be installed during setup steps.

Bear in mind that training the model might take a lot of time depending on the type of machine you are using for training and if you're using GPU or CPU.

```
python FasterRCNN.py
```

**TIP:** If you don't own any machine with heavy GPU you can use one of the ready to go [Data Science Virtual Machine images in Azure](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.windows-data-science-vm).

When the training and evaluation will be completed, you should see something similar to this:

```
Evaluating Faster R-CNN model for 20 images.
Number of rois before non-maximum suppression: 550
Number of rois  after non-maximum suppression: 87
AP for            sink = 0.4429
AP for          pillow = 0.1358
AP for          toilet = 0.8095
AP for            lamp = 0.5404
AP for         curtain = 0.7183
AP for           towel = 0.0000
AP for             tap = 0.1111
AP for             bed = 0.8333
Mean AP = 0.4489
```

Trained model, neural network topology and evaluated images (with plotted results) can later be found in `Output` folder located in [Detection/FasterRCNN](Detection/FasterRCNN).
