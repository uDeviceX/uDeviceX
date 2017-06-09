dhost () ( # dispatch hostname
    h=`hostname -a`
    awk '
       BEGIN {
	   h = ARGV[1]
	   if (h ~ /^daint/)  print "daint"
	   else               print "panda"
       }
    ' "$h"
)
