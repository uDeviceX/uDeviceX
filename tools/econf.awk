#!/usr/bin/awk -f

# Manipulate kernel execution configurations

function trim(s) {
    sub(/^[ \t]*/, "", s)
    sub(/[ \t]*$/, "", s)
    return s
}

function uncom(l) { # skip comments
    sub(/\/\/.*/, "", l)
    return l
}

function kernp(l) {
    l = uncom(l)
    return l ~ /<<<.*>>>/
}

function spl(l) { # split line into: b(efor)<<<m(id)>>a(fter)
    b = l; sub(/<<<.*/, "", b)
    a = l; sub(/.*>>>/, "", a)
    m = l; sub(/>>>.*/, "", m); sub(/.*<<</, "", m)
}

function removeNs(m,   n, args) {
    n = split(m, args, ",") # TODO: assumes no <<<f(a, b), c>>>
    if (n == 2) return m
    
    Dg = args[1]; Db = args[2]; Ns = args[3]
    Dg = trim(Dg); Db = trim(Db); Ns = trim(Ns)
    
    if (Ns != 0) return m
    return Dg ", " Db
}

kernp($0) {
    l0 = $0
    
    spl(l0) # split
    m = removeNs(m)

    l = b "<<<" m ">>>" a
    print "==", l0
    print "  ", l
}
