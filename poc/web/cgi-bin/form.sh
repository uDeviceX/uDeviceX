#!/bin/bash
echo "Content-type: text/html"
echo ""
echo '<html>'
echo '<head>'
echo '<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">'
echo '<title>Form Example</title>'
echo '</head>'
echo '<body>'
echo "<form method=GET action=\"${SCRIPT}\">"\
     '<table nowrap>'\
     '<tr><td>Input</TD><TD><input type="text" name="val_x" size=12></td></tr>'\
     '<tr><td>Section</td><td><input type="text" name="val_y" size=12 value=""></td>'\
     '</tr></table>'
echo '<input type="radio" name="val_z" value="1" checked> Option 1<br>'\
     '<input type="radio" name="val_z" value="2"> Option 2<br>'\
     '<input type="radio" name="val_z" value="3"> Option 3'
echo '<br><input type="submit" value="Process Form">'\
     '<input type="reset" value="Reset"></form>'

if test ! -z "$QUERY_STRING"
then
    XX=`echo "$QUERY_STRING" | sed -n 's/^.*val_x=\([^&]*\).*$/\1/p' | sed "s/%20/ /g"`
    YY=`echo "$QUERY_STRING" | sed -n 's/^.*val_y=\([^&]*\).*$/\1/p' | sed "s/%20/ /g"`
    ZZ=`echo "$QUERY_STRING" | sed -n 's/^.*val_z=\([^&]*\).*$/\1/p' | sed "s/%20/ /g"`
    echo "val_x: " $XX
    echo '<br>'
    echo "val_y: " $YY
    echo '<br>'
    echo "val_z: " $ZZ
fi
echo "q: $QUERY_STRING" >&2
echo '</body>'
echo '</html>'
