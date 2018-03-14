#!/usr/bin/env octave-qf

1;

function r = ac_bc_cross(a, b, c)
  r = cross(a - c, b - c)
endfunction

a = [1, 2, 3]
b = [4, 5, 6]
c = [7, 8, 9]
