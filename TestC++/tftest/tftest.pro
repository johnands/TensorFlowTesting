TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt
INCLUDEPATH += /home/johnands/QTensorFlow/include

LIBS += -L/home/johnands/QTensorFlow/lib64 -ltensorflow

SOURCES += main.cpp

