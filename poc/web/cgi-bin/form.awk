#!/usr/bin/awk -f

BEGIN {
    q = ENVIRON["QUERY_STRING"]

    print "Content-type: text/html"
    print
    print "<form>"
    print  "<input name=\"y\" value=\"1\">"
    print  "<input name=\"z\" value=\"10\">"
    print "<input type=\"submit\">"
    print  "</form>"

    print "Q:" q
}
