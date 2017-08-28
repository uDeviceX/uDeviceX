# start dir
s=strt

# base dir
b=$SCRATCH/re

if test -z "$SCRATCH"
then
    printf 'u.re: $SCRATCH is not set%s\n' >&2
    exit
fi

mkdir -p "$b"

if test ! -d "$b"
then
    printf 'u.re: cannot create %s\n' "$b" >&2
    exit
fi
