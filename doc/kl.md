# kernel launch macros

All but `unsafe` skip kernels with zero thread number.

Examples:

    KL(fun, (i+1, j, k), ())     -> fun<<<i+1, j, k>>>()
	KL(fun, (i+1, j, k), (a, b)) -> fun<<<i+1, j, k>>>(a, b)

## release
## trace
## peek
## unsafe
