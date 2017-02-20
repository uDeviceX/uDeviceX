from vcustom import *
from visit   import *

def vec():
    DefineVectorExpression("v2d", "{u, v}")
    OpenDatabase("../h5/av2d.xmf")
    AddPlot("Vector", "v2d", 0, 0)
    atts = VectorAttributes()
    atts.useStride = 1
    atts.scaleByMagnitude = 0
    SetPlotOptions(atts)
    AddOperator("Elevate", 0)

def rbc():    
    OpenDatabase("v.visit")
    AddPlot("Subset", "PLY_mesh", 0, 0)

def wall():
    OpenDatabase("../h5/wall.xmf")
    AddPlot("Pseudocolor", "wall", 0, 0)
    AddOperator("Isosurface", 0)

def view():
    vv = GetView3D()
    vv.viewNormal = (-0.0468824, 0.0152569, 0.998784)
    vv.viewUp = (-0.0142489, 0.999771, -0.0159408)
    SetView3D(vv)

wall()
#vec()
rbc()
view()
DrawPlots()

go_last()


lst = TimeSliderGetNStates()-1
s=10
for p in range(0, 100, s):
    time = int(lst*p/100.0)
    SetTimeSliderState(time)
    fn = "visit.%03i" % p
    sw(fn)
