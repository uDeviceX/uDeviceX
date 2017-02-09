#!/usr/bin/awk -f

# A tool for manual dependencies tracing
#
# Usage:
# ../tools/deps.awk *.cu

BEGIN {
    asplit("cuda.h stdlib.h stdio.h errno.h mpi.h H5Part.h hdf5.h pthread.h " \
	   "cuda_runtime.h stdint.h unistd.h math.h math_functions.h " \
	   "geom-wrapper.h helper_math.h", sys_hdr)
}

function dep_list(d, f, n,   i, ans, sep) { # return dep as a string
    ans = ""
    for (i = 1; i <= n; i++) {
	sep = i == 1 ? "" : " "
	ans = ans sep d[f,i]
    }
    return ans
}

function dswap(d, f, i, j,     tmp) {
    tmp = d[f, i]; d[f, i] = d[f, j]; d[f, j] = tmp
}

function dsort(d, f, n,   i, j) { # sort deps in O(N^2)
    for (i=1; i<=n; i++)
	for (j=i; j<=n; j++)
	    if (d[f,j] < d[f,i]) dswap(d, f, i, j)
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
    # lesser and greater brackets
    has_lg = sub(/^</, ""); sub(/>$/, "") # trim `<>'

    has_h =     $0 ~ /[.]h$/
    has_sys =   $0 ~ /^sys\//
    has_thrust = $0 ~ /^thrust\//

    if (!has_h || has_sys || has_lg) next
    if ($0 in sys_hdr)                   next

    n = ++ndep_be[FILENAME  ] # `f' depends on FILENAME (shouble be "before")
    dep_be [FILENAME,n] = $0

    n = ++ndep_af[$0  ]       # FILENAME depends of `f' (shouble be "after")
    dep_af [$0,n] = FILENAME

    pos_be[FILENAME,$0] = FILENAME ":" FNR
    pos_af[$0,FILENAME] = FILENAME ":" FNR

}

function emacs_buffer() {
    for (f in ndep_be) {
	hdr = f ~ /[.]h$/ # is it a header
	if (!hdr) continue

	for (i = 1; i <= ndep_be[f]; i++) { # format a messages for
					    # emacs *compilation*
					    # buffer
	    printf "%s: error: %s\n", pos_be[f, dep_be[f,i]], dep_be[f,i]
	    for (j = 1; j <= ndep_af[f]; j++)
		printf "%s: warning: %s\n", pos_af[f, dep_af[f,j]], dep_af[f,j]
	}
	print "\n"
    }
}

function makefile_header() {
    printf "%s", "# Generated with by ../tools/deps.awk *.cu *.h\n"
    printf "%s", "# (do not include '*.h' files in other '*.h' files)\n"
    printf "\n"
}

function makefile_rule() {
    for (f in ndep_be)
	dsort(dep_be, f, ndep_be[f])

    makefile_header()
    for (f in ndep_be)
	printf "%s: %s\n", f, dep_list(dep_be, f, ndep_be[f]) | "sort"
 }

END {
    emacs_buffer()

    makefile_rule()
}
