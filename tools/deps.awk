#!/usr/bin/awk -f

# A tool for manual dependencies tracing
#
# Usage:
# ../tools/deps.awk *.cu

BEGIN {
    asplit("cuda.h stdlib.h stdio.h errno.h mpi.h H5Part.h hdf5.h pthread.h " \
	   "cuda_runtime.h stdint.h unistd.h math.h math_functions.h", sys_hdr)
}

function asplit(str, arr,   temp, i, n) {  # make an assoc array from str
    n = split(str, temp)
    for (i = 1; i <= n; i++) arr[temp[i]]++
    return n
}

/^#include/ {
    sub(/^#include/, "")
    sub(/^[\t ]*/, ""); sub(/[\t ]*$/, "") # trimp spaces and tabs
    sub(/^"/, ""); sub(/"$/, "") # trim `"'
    # lg_br: lesser and greater brackets
    lg_br = sub(/^</, ""); sub(/>$/, "") # trim `<>'

    has_h =     $0 ~ /[.]h$/
    has_sys =   $0 ~ /^sys\//
    has_thrust = $0 ~ /^thrust\//

    if (!has_h || has_sys || has_thrust) next
    if ($0 in sys_hdr)                   next

    dep[FILENAME]=dep[FILENAME] " " $0
}

END {
    for (f in dep)
	print f ":" dep[f]
}
