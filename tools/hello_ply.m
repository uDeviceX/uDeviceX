#!/usr/bin/env octave-qf

% Read coordinates and faces

% Usage (octave-qf wrapper should be on the PATH):
% > ./hello_ply.m test_data/rbc.ply

fn  = argv(){1}; % file name
% fn  = 'test_data/rbc.ply';

nvar  = 6; % x, y, z, u, v, w
nv_pf = 3; % number of vertices per face (3 for triangle)

fd = fopen(fn); nl  = @() fgetl(fd); % next line

nl(); nl();
l = nl(); nv = sscanf(l, 'element vertex %d');
nl(); nl(); nl(); nl(); nl(); nl();
l = nl(); nf = sscanf(l, 'element face %d');
nl(); nl();

D = fread(fd, [nvar     , nv], 'float32');
i = 1;  xx = D(i++, :);  yy = D(i++, :);  zz = D(i++, :);

D = fread(fd, [nv_pf + 1, nf], 'int32');
i = 2; f1 = D(i++, :); f2 = D(i++, :); f3 = D(i++, :); % skip the
                                                       % first
fclose(fd);

dlmwrite(stdout() , [xx', yy', zz'], ' ');
