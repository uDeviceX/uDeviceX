#!/bin/sh

echo "Content-type: text/html"
echo
echo '<form action="form0.sh">'
echo '<input name="x">'
echo '<input type="submit">'
echo '</form>'
echo "q: $QUERY_STRING"
