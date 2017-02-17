#!/usr/bin/env octave-qf

% Usage (octave-qf wrapper should be on the PATH):
% > ./hello.m test_data/ff.h5

% octave hello world
fn  = argv(){1}; % file name
X = 1; Y = 2; Z = 3;
sq  = @squeeze;
grd = @(n) (1:n) - 1/2;

load(fn, 'u', 'v', 'w');
vx = sq(u); vy = sq(v); vz = sq(w);
nx = size(vx, X); ny = size(vy, Y); nz = size(vz, Z);
xx = grd(nx); yy = grd(ny); zz = grd(nz);

reduce_12 = @(f, d1, d2) sq(mean(mean(f, d1), d2));
reduce_xz = @(f) reduce_12(f, X, Z);

vx = reduce_xz(vx);

% dlmwrite('vx.dat', [yy' vx'], ' ');
dlmwrite(stdout, [yy' vx'], ' ');
