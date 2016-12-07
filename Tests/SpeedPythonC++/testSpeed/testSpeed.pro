TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    math/random.cpp \
    ../../../../FYS4460-MD/math/activationfunctions.cpp

INCLUDEPATH += /home/johnands/QTensorFlow/include
LIBS += -L/home/johnands/QTensorFlow/lib64 -ltensorflow

LIBS += -larmadillo -lblas -llapack

release {
    DEFINES += ARMA_NO_DEBUG
    QMAKE_CXXFLAGS_RELEASE -= -O2
    QMAKE_CXXFLAGS_RELEASE += -O3
}

HEADERS += \
    math/random.h \
    ../../../../FYS4460-MD/math/activationfunctions.h
