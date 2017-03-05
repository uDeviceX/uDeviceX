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

function reorder128(m,   n, fst, scnd, args) {
    trans = 0

    n = split(m, args, ",") # TODO: assumes no <<<f(a, b), c>>>
    if (n != 2) return m

    fst = args[1]; scd = args[2]
    fst = trim(fst); scd = trim(scd)
    if (scd != "128") return m

    n = split(fst, args, "/")
    if (n != 2) return m
    fst = args[1]; scd = args[2]
    fst = trim(fst); scd = trim(scd)
    if (scd != "128") return m

    sub(/^[(]/, "", fst)
    sub(/[)]$/, "", fst)

    n = split(fst, args, "+")
    if (n != 2) return m
    fst = args[1]; scd = args[2]
    fst = trim(fst); scd = trim(scd)

    m = sprintf("k_cnf(%s)", fst)
    trans = 1
    return m
}


kernp($0) {
    l0 = $0

    spl(l0) # split
    m = reorder128(m)

    l = b "<<<" m ">>>" a

    print trans ? l : l0
    next
}

{
    print
}
