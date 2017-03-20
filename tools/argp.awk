#!/usr/bin/awk -f

function init() {
    # default values (change here)
    defs = \
	"RBCnv 498 "                    \
	"RBCnt 992 "                    \
	"doublepoiseuille false "		\
	"hdf5field_dumps false "		\
	"hdf5part_dumps false "			\
	"pushtheflow false "			\
	"rbcs false "				\
	"steps_per_dump 1000 "			\
	"steps_per_hdf5dump 2000 "		\
	"tend 50 "				\
	"wall_creation_stepid 5000 "		\
	"walls false "
}

function kvsplit(s, kv,    n, t, i) { # key-value split of a string
    n = split(s, t)
    for (i = 1; i<=n; i+=2)
	kv[t[i]] = t[i+1]
    return n
}

function keyp(k) { # key starts with a minus
    return k ~ /^-/
}

# key and value
function getk(s) {sub(/^-/, "", s); sub(/=.*/, "", s); return s}
function getv(s) {sub(/^-/, "", s); sub(/.*=/, "", s); return s}

function err(c) {
    printf "(argp) cannot parse: \"%s\"\n", c | "cat 1>&2"
    exit(1)
}

function boolp(k) { # not `=' means boolean
    return k !~ /=/
}

BEGIN {
    init()
    kvsplit(defs, kv_default)
    kvsplit(defs, kv)

    iarg = 1 #
    file = ARGV [iarg++]

    while (iarg < ARGC)  {
	cur = ARGV[iarg++]
	if (!keyp(cur)) err(cur)
	k = getk(cur)
	if (boolp(cur)) kv[k] = "true"
	else            kv[k] = getv(cur)
    }
    ARGC = 2 # to read the first argument as a file

    printf "/* Part I (%s) */\n\n", file
}

{ # print the file
    print
}

function max(a, b) {return a > b ? a : b}

END {
    for (k in kv) { # collect maximum length for formatting
	v = kv[k]
	sk = max(length(k), sk)
	sv = max(length(v), sv)
    }

    # dump output
    printf "\n/* Part II (added by tools/argp) */\n\n", FILENAME
    u_tmpl = sprintf("#undef  %%%ss", sk)
    d_tmpl = sprintf("#define %%%ss %%%ss", sk, sv + 2) # with brackets

    for (k in kv) {
	line = sprintf(u_tmpl, k)
	print line | "sort"
    }
    close("sort")
    printf "\n"

    for (k in kv) {
	v = kv[k]; vd = kv_default[k]; vp = "(" v ")"
	line = sprintf(d_tmpl, k, vp)
	if (v != vd) line = line "    /* */" # marking non-default
					     # value
	print line | "sort"
    }
    close("sort")
}
