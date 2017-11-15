# coding conventions

## naming

* names of local variables are short and should be understandable from the context
* names of global variables should be as descriptive as possible
* arrays of simple structures `x` are names `xx`, eg. `pp` is an array of particles, `cc` is an array of colors, `ii` is an array of indices

## file structure

* no include guards; no "headers in headers" if easily avoidable.
* all cuda kernels should be coded in a separate header `.h` file.
