function [tri, xx, yy, zz] = u_off_read(path)
  X = 1; Y = 2; Z = 3;
  f = fopen(path, 'r');
  if f == -1
    error('not a file "%s"', path);
  endif

  s = fgets(f); % "OFF"
  if strncmp(s, "OFF", 3) == 0
    error('not an off file "%s"', path)
  endif

  s = fgets(f);
  [D, cnt] = sscanf(s, "%d %d %d");
  if cnt != 3
    error('wrong line: "%s" in "%s"', s, path)
  endif
  nv = D(1); nt = D(2);

  [D, cnt] = fscanf(f, '%f %f %f', [3, nv]);
  if cnt != nv * 3
    error('fail to read xyz in "%s"', path)
  endif

  xx = D(X, :); yy = D(Y, :); zz = D(Z, :);

  [D, cnt]  = fscanf(f,'%d %d %d %d\n', [4, nt]);
  if cnt != nt * 4
    error('failt to read triangles in "%s"', path)
  endif

  tri = D(2:end, :)';
  tri = tri + 1;
  if fclose(f) != 0
    error('fail to close "%s"', path)
  endif
endfunction
