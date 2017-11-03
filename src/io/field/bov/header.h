void write_header(const char *fhname, const char *fdname, int Lx, int Ly, int Lz, int ncmp, const char varname) {
    FILE *f;
    long x, y, z;
    
    x = m::coords[0] * Lx;
    y = m::coords[1] * Ly;
    z = m::coords[2] * Lz;

    f = safe_open(fhname, "w");
    
    fprintf(f, "DATA_FILE: %s\n", fdname);
    fprintf(f, "DATA_SIZE: %ld %ld %ld\n", x, y, z);
    fprintf(f, "DATA_FORMAT: FLOAT\n");
    fprintf(f, "VARIABLE: %s\n", varname);
    fprintf(f, "DATA_ENDIAN: LITTLE\n");
    fprintf(f, "CENTERING: zonal\n");
    fprintf(f, "BRICK_ORIGIN: %g %g %g\n", 0, 0, 0);
    fprintf(f, "BRICK_SIZE: %g %g %g\n", (double) x, (double) y, (double) z);
    fprintf(f, "DATA_COMPONENTS: %d\n", ncmp);
    
    fclose(f);
}
