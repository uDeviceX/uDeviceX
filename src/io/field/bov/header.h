void write_header(const char *fhname, const char *fdname, int nx, int ny, int nz, int ncmp, const char *varname) {
    FILE *f;
    long Lx, Ly, Lz;
    
    Lx = m::coords[0] * nx;
    Ly = m::coords[1] * ny;
    Lz = m::coords[2] * nz;

    UC(efopen(fhname, "w", /**/ &f));
    
    fprintf(f, "DATA_FILE: %s\n", fdname);
    fprintf(f, "DATA_SIZE: %ld %ld %ld\n", Lx, Ly, Lz);
    fprintf(f, "DATA_FORMAT: FLOAT\n");
    fprintf(f, "VARIABLE: %s\n", varname);
    fprintf(f, "DATA_ENDIAN: LITTLE\n");
    fprintf(f, "CENTERING: zonal\n");
    fprintf(f, "BRICK_ORIGIN: %g %g %g\n", 0., 0., 0.);
    fprintf(f, "BRICK_SIZE: %g %g %g\n", (double) Lx, (double) Ly, (double) Lz);
    fprintf(f, "DATA_COMPONENTS: %d\n", ncmp);
    
    UC(efclose(f));
}
