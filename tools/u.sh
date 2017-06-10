# utils

inc ()  { #
          # inc HOST.file
          #   source `daint.file' on daint and `panda.file' on panda/falcon
    . `inc0 "$1"`
}

dhost () { # dispatch hostname
    awk '
       BEGIN {
	   h = ARGV[1]
	   if (h ~ /^daint/)  print "daint"
	   else               print "panda"
       }
    ' `hostname`
}


inc0 () {
    awk '
       BEGIN {
	   h = ARGV[1]; f = ARGV[2]
	   gsub(/HOST/, h, f)
	   print f
       }
    ' `dhost` "$1"
}

