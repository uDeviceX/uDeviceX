#!/usr/bin/env octave-qf

1;

function r = ac_bc_cross(a, b, c)
  r = cross(a - c, b - c);
endfunction

a = [1, 3, 2];
b = [6, 7, 8];
c = [9, 10, 13];

ac_bc_cross(a, b, c)
