# Machine Learning Engineer Nanodegree
# Capstone Project
## Project: A-Z Handwritten Character Recognizer

## Project Overview

My project's aim is to program the computer to identify hand-written alphabets via matrix
operations. Each alphabet image contains 28*28 pixels, and we create a matrix using these
pixels. By multiply the matrix to several sample matrixes, the pixels are converted into a deep neural network. And finally, we employ adam optimizer method so that the computer can predict the highest possibility of the alphabet written.

Keywords: deep learning, ANN, Feature Extraction, CNN, English, Machine Recognition, natural
and physical sciences, image data, image processing

## Problem Statement
The main objective of this research is to find a new solution for handwritten text recognition of
different fonts and styles by improving the design structure of the traditional Artificial Neural
Network (ANN). ANNs have been successfully applied to pattern recognition, association and
classification, forecast studies, and control applications, to name a few. The recognition results of
such text or handwritten materials are then fed into Optical Character Recognition (OCR) as an
electronic translation of images of handwritten, typewritten or printed text into machine-editable
text. OCR is a field of research that is fully developed and has been quite useful in pattern
recognition, artificial intelligence and machine vision. Consequently, typewritten text recognition
that is void of any distortions is now considered largely a solved problem. However, the direct use
of OCR on handwritten characters remains a very difficult problem to resolve, yielding extremely
low reading accuracy. handwritten document recognition is currently a difficult problem; as
different people have different handwriting styles. Scanning, segmentation and classification are
the general processes that are being used to recognize handwritten documents. ANNs have
proven to be excellent recognizers of printed characters and handwritten characters.

## Domain Background
Character recognition is one of the most important research fields of image processing and
pattern recognition. Character recognition is generally known as Optical Character Recognition
(OCR).OCR is the process of electronic translation of handwritten images or typewritten text into
machine editable text. It becomes very difficult if there are lots of paper based information on
companies and offices. Because they want to manage a huge volume of documents and records.
Computers can work much faster and more efficiently than human. It is used to perform many of
the tasks required for efficient document and content management. But computer knows only
alphanumeric characters as ASCII code. So computer cannot distinguish character or a word from
a scanned image. In order to use the computer for document management, it is required to
retrieve alphanumeric information from a scanned image. There are so many methods which are
currently used for OCR and are based on different languages. The existing method like Artificial
Neural Network (ANN) based on English Handwritten character recognition needs the features to
be extracted and also the performance level is low. So a Convolutional Neural Network (CNN)
based English handwritten character recognition method is used. It's a deep machine learning
method for which it doesn't want to extract the features and also a fast method for character recognition.

My personal **motivation** is that i Have faced a problem, where i wanted to transform a paper
document into a digital one, but i had to type it character by character. i have been trying to find an
easy way to solve the problem,simply The purpose of our project is to recognize hand-written
alphabets, so the computer can automatically identify the characters without any manual input.
The link to my datasource is:

(https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format) 

### Requirement
### Install

This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [seaborn](https://seaborn.pydata.org/)
- [tensorflow](https://www.tensorflow.org/)
- [keras](https://keras.io/)


You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. 

### Code

code is provided in the `A-Z Handwritten Character recognizer.ipynb` notebook file. the `A_Z Handwritten Data.csv` dataset file to review the project.

### Run

In a terminal or command window, navigate to the top-level project directory `A-Z Handwritten Character recognizer/` (that contains this README) and run one of the following commands:

```bash
ipython notebook A-Z Handwritten Character recognizer.ipynb
```  
or
```bash
jupyter notebook A-Z Handwritten Character recognizer.ipynb
```

This will open the Jupyter Notebook software and project file in your browser.

### Data

The A_Z Handwritten dataset contains 26 folders (A-Z) containing handwritten images in size 28*28 pixels, each alphabet in the image is centre fitted to 20*20 pixel box.Each image is stored as Gray-level.
The images manily are taken from NIST(https://www.nist.gov/srd/nist-special-database-19)

**you can find the dataset here** (https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format) 

## Solution Statement
A Convolutional Neural Network (CNN) is a special type of feed-forward multilayer trained in
supervised mode. The CNN trained and tested our database that contains 372451 of
handwritten english characters. 

## Evaluation Metrics
Generating a confusion matrix,for summarizing the performance of a classification algorithm.
Classification accuracy alone can be misleading if you have an unequal number of observations
in each class or if you have more than two classes in your dataset. Calculating a confusion matrix
can give you a better idea of what your classification model is getting right and what types of
errors it is making.

