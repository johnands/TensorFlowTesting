TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt
INCLUDEPATH += /home/johnands/Documents/TensorFlow/tensorflow/bazel-genfiles
INCLUDEPATH += /home/johnands/Documents/TensorFlow/tensorflow/bazel-tensorflow
INCLUDEPATH += /usr/local/lib/python2.7/dist-packages/tensorflow/include

LIBS += -L/home/johnands/Documents/TensorFlow/tensorflow/bazel-bin/tensorflow -ltensorflow

SOURCES += main.cpp

