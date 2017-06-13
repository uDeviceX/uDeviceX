#!/bin/bash

# udx commands

msg () { printf '%s\n' "$@" | cat 1>&2; }

if test $# = 1 -a "$1" = -h; then msg 'udx make wrapper'; exit; fi

h=`u.host`

. u.load."$h"
make "$@"
