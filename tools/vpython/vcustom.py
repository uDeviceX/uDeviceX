import sys
import os
from visit import *

def annoff():
    '''hide annotation'''
    a = AnnotationAttributes()
    a.axes2D.visible = 0
    a.axes3D.visible = 0
    a.axes3D.triadFlag = 0
    a.axes3D.bboxFlag = 0
    a.userInfoFlag = 0
    a.databaseInfoFlag = 0
    a.legendInfoFlag = 0
    SetAnnotationAttributes(a)
    
def sf(fn = "script.py" ):
    ''' save visit state to a python file '''
    print "(vcustom.py) saving: %s" % fn
    f = open(fn, "wt")
    WriteScript(f)
    f.close()

def v():
    ''' run visit gui '''
    OpenGUI("-nosplash", "-noconfig")

def sw(fn = "visit", z = 1):
    ''' save png image '''
    a = GetSaveWindowAttributes()
    a.family = 0
    a.outputToCurrentDirectory = 1
    h = int(1024*z); w = int(1.2*h)
    a.width  = w
    a.height = h
    a.quality = 100
    a.fileName = fn
    a.format = a.PNG
    a.screenCapture = 0
    a.resConstraint = a.NoConstraint
    SetSaveWindowAttributes(a)
    fn = SaveWindow()
    print "(vcustom.py) saving %s\n" % fn

def go_last():
    '''got to the last slide'''
    lst = TimeSliderGetNStates()-1
    SetTimeSliderState(lst)

del sys.argv[0] # remove vcustom.py from agrument list
Launch()
execfile(sys.argv[0])
