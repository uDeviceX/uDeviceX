# map

common mapping routines and structure

## Map structure:

general structure for `n` sets of objects  
let `stride = NFRAGS + 1`:  
`counts` and `starts` should have size at least `n * stride`  
`offsets` should have size at least `(n+1) * stride`  
