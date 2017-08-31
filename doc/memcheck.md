# memory check

if `MEM` is set `udx` is ran with cuda-memcheck
and `MEM` is used as a list of parameters

    MEM= u.test test/*
	MEM="‐‐leakcheck ‐‐blocking"              u.test test/*


if `VAL` is set `udx` is ran with valgrind and `VAL` is used as a list
of parameters. Works only on panda.

    VAL= u.test test/*
    VAL="--leak-check=full --show-leak-kinds=all"  u.test test/*
