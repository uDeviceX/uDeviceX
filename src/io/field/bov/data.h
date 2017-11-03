void write_values(const char *fdname, int nx, int ny, int nz, const float *data) {
    long n;
    f = safe_open(fdname, "wb");

    n = nx * ny * nz;
    fwrite(d, sizeof(float), n, f);
    
    fclose(f);
}
