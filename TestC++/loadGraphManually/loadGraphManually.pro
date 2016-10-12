TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

LIBS += -larmadillo -lblas -llapack

SOURCES += main.cpp \
    neuralnetwork.cpp \
    activationfunctions.cpp

HEADERS += \
    neuralnetwork.h \
    activationfunctions.h

