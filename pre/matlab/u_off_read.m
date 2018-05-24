function [tri, xx, yy, zz] = u_off_read(path)
  f = fopen(name, 'r');
  s = fgets(f);

  if ~scmp(s(1:3), 'OFF')
    error('not an off file "%s"', path);
  end

  s = fgets(f);
  [a, s] = stok(s); nvert = s2num(a);
  [a, s] = stok(s); nface = s2num(a);

  [A,cnt] = fscanf(f,'%f %f %f', 3*nvert);
  if cnt~=3*nvert
    warning('Problem in reading vertices.');
  end
  A = reshape(A, 3, cnt/3);
  vertex = A;

  [A,cnt] = fscanf(f,'%d %d %d %d\n', 4*nface);
  if cnt~=4*nface
    warning('Problem in reading faces.');
  end
  A = reshape(A, 4, cnt/4);
  face = A(2:4,:)+1;
  fclose(f);
endfunction
