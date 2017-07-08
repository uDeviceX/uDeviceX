#!/usr/bin/awk -f

BEGIN {
    q = ENVIRON["QUERY_STRING"]
    
    print "Content-Type: text/html"
    print
    print "<html>"
    print  "<head><title>Hello from awk</title></head>"
    print "<body>"

    print "<P>"
    print "There are several types of search:"
    print "<SELECT NAME=\"type\">"
    print "<OPTION>Case Insensitive Substring Match"
    print "<OPTION>Exact Match"
    print "<OPTION>Case Sensitive Substring Match"
    print "<OPTION>Regular Expression Match"
    print "</SELECT>"
    print "<P>"

    print  "<form method=GET action=\"form.awk\">"
    print  "<br><input type=\"submit\" value=\"Post\"></form>"
    print "q:", q, length(q)

    print "</body></html>"
}
