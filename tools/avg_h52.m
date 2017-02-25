#!/usr/bin/env octave-qf

% Usage (octave-qf wrapper should be on the PATH):
% > ./avg_h5.m test_data/ff.h5

fn  = argv(){1}; % file name
X = 1; Y = 2; Z = 3;
sq  = @squeeze;
grd = @(n) (1:n) - 1/2;

load(fn, 'u', 'v', 'w');
vx = sq(u); vy = sq(v); vz = sq(w);
nx = size(vx, X); ny = size(vy, Y); nz = size(vz, Z);
xx = grd(nx); yy = grd(ny); zz = grd(nz);

vx = sq(mean(vx, 1));
vx = sq(mean(vx, 2));

% dlmwrite('vx.dat', [zz' vx'], ' ');
dlmwrite(stdout, vx, ' ');
