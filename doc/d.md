# generic [d]evice API

The goal is to make cuda API and kernel functions generic. Meaning
they can be replaced by CPU calls.

generic files of the interface
* `src/d/api.h` API calls
* `src/d/q.h` function and variable type qualifiers
* `src/d/ker.h`

Generic device can by implemented by `cuda` and by `cpu`
* `src/d/cuda`
* `src/d/cpu`

if `DEV_CUDA` defined it is `cuda`, if `DEV_CPU` is defined it is
`cpu`.
