#!/bin/bash

# udx commands

msg () { printf '%s\n' "$@" >&2; }

if test $# = 1 -a "$1" = -h; then msg 'udx make wrapper'; exit; fi

h=`u.host`

. u.load."$h"
. u.make."$h"
