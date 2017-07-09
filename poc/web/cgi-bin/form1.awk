#!/usr/bin/awk -f

BEGIN {
    load()
    print "Content-type: text/html"
    print
    print "<form>"
    input("x")
    print  "<input name=\"y\" value=\"2\">"
    print  "<input name=\"z\" value=\"2\">"
    print "<input type=\"submit\">"
    print  "</form>"
    kv()
    print "sum = ", V["x"] + V["y"] + V["z"]
    save()
}

function input(k,   v) {
    v = V[k]
    print "v: ", v
    printf "<input name=%s value=%s>\n", q(k), q(v)
}

function q(e) { return "\"" e "\"" }

function load(   l) {
    while (getline l < ".V" > 0)
	load_one(l)
}

function load_one(l,   a, k, v) {
    split(l, a, "\t")
    k = a[1]; v = a[2]
    V[k] = v
}

function save(   k, v) {
    for (k in V) {
	v = V[k]
	print k "\t" v > ".V"
    }
}

function kv(   q, a, n) { # sets `V'
    q = ENVIRON["QUERY_STRING"]
    n = split(q, a, "&")
    kv0(a)
}

function kv0(a,  i) {
    for (i = 1; i in a; i++) kv_one(a[i])
}

function kv_one(p,   n, a, k, v) {
    split(p, a, "=")
    k = a[1]; v = a[2]
    V[k] = v
}
