dhost () ( # dispatch hostname
    h=`hostname`
    awk '
       BEGIN {
	   h = ARGV[1]
	   if (h ~ /^daint/)  print "daint"
	   else               print "panda"
       }
    ' "$h"
)
