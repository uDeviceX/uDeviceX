namespace label {
enum {
    BULK,  /* remains in bulk */
    WALL,  /* becomes wall particle */
    DEEP   /* deep inside the wall */
};
void dev(const Sdf *sdf, int n, const Particle*, /**/ int *labels);
void hst(const Sdf *sdf, int n, const Particle*, /**/ int *labels);
}
