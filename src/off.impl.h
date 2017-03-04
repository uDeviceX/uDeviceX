namespace off {
  /* off files

  [1] https://en.wikipedia.org/wiki/OFF_(file_format)

  */

  /* return faces: f0[0] f1[0] f2[0]   f0[1] f1[1] ... */
  void f2faces(const char *f, int* faces) {
    char buf[1024];
    FILE *fd = fopen(f, "r");
    fgets(buf, sizeof buf, fd); /* skip OFF */

    int nv, nf;
    fscanf(fd, "%d %d %*d", &nv, &nf); /* skip `ne' and all vertices */
    for (int iv = 0; iv < nv;  iv++) fscanf(fd, "%*e %*e %*e");

    int ifa = 0, ib = 0;
    for (/*   */ ; ifa < nf; ifa++) {
      int f0, f1, f2;
      fscanf(fd, "%*d %d %d %d", &f0, &f1, &f2);
      faces[ib++] = f0; faces[ib++] = f1; faces[ib++] = f2;
    }
    fclose(fd);
  }

  /* return vertices */
  void f2vert(const char *f, float* vert) {
    char buf[1024];
    FILE *fd = fopen(f, "r");
    fgets(buf, sizeof buf, fd); /* skip OFF */

    int nv;
    fscanf(fd, "%d %*d %*d", &nv); /* skip `nf' and `ne' */
    int iv = 0, ib = 0;
    for (/*   */ ; iv < nv;  iv++) {
      float x, y, z;
      fscanf(fd, "%e %e %e", &x, &y, &z);
      vert[ib++] = x; vert[ib++] = y; vert[ib++] = z;
    }

    fclose(fd);
  }

} /* namespace off */
