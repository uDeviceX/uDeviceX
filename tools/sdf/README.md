# SDF tools

## Intro
SDF format:
ex ey ez
nx ny nz
<D>

ex, ey, ez: extends ASCII
nx, ny, nz: sizes   ASCII
D: a block of nx*ny*nz floating point numbers (x is fastest and z is
slowest index). BINARY

Grid is cell central-ed
* s = e/n : step size
* o = s/2 : origin
* r(i) = o + i*s : coordinate from index 0 .. n - 1

## Utils
* [sdf.2bov](sdf.2bov) convert to bov (format visit cat read)
* [sdf.2per](sdf.2per) make periodic by two reflections
* [sdf.2txt](sdf.2txt) convert to ad hoc text format
* [sdf.cut](sdf.cut) cut file using low and high indices in each
  dimension
* [sdf.filter](sdf.filter) filters stream of positions based on the level of sdf
* [sdf.reflect](sdf.reflect) reflect against a boundary (">" becomes "><")
* [sdf.shuffle](sdf.shuffle) shuffle dimensions (xyz -> yxz)
* [sdf.smooth](sdf.smooth) smooth using radial basic function
