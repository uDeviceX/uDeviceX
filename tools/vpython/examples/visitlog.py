# Visit 2.11.0 log file
ScriptVersion = "2.11.0"
if ScriptVersion != Version():
    print "This script is for VisIt %s. It may not work with version %s" % (ScriptVersion, Version())
visit.ShowAllWindows()
OpenDatabase("globe.silo", 0)
# The UpdateDBPluginInfo RPC is not supported in the VisIt module so it will not be logged.
AddPlot("Pseudocolor", "u", 1, 0)
DrawPlots()
