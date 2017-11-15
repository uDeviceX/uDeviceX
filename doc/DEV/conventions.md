# coding conventions

## style

for emacs users: see [cstyle/emacs.md](cstyle/emacs.md)

## naming

### variables

* names of local variables are short and should be understandable from the context
* names of global variables should be as descriptive as possible
* arrays of simple structures `x` have names `xx`, eg. `pp` is an array of particles, `cc` is an array of colors, `ii` is an array of indices

### functions

* a function name should be descriptive on its own or inside its namespace
* arguments are ordered as follow:
  * input is at the beginning
  * input/output come after input and start with `/* io */`
  * output comes after input/output and start with `/**/` or `/* o */` depending on the context
  * workspace comes at the end ans starts with `/* w */`

## file structure

* no include guards; no "headers in headers" if easily avoidable.
* all cuda kernels should be coded in a separate header `.h` file.
* a module is implemented inside its own folder `A`
  * it has its own object `A/imp.cu` or `A/imp.cpp`
  * interface `A/imp.h`
  * implementation can be done in separate files inside `A/imp/` folder
  * cuda code should be inside `A/dev.h` or, if multiple files, `A/dev/` folder
* modules can have submodules, which follow the same structure as above, e.g. submodule `B` inside module `A` belongs to `A/B/` folder


