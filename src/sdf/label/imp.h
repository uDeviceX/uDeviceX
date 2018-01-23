namespace label {
enum {
    LABEL_BULK,  /* remains in bulk */
    LABEL_WALL,  /* becomes wall particle */
    LABEL_DEEP   /* deep inside the wall */
};
void dev(const Sdf *sdf, int n, const Particle*, /**/ int *labels);
void hst(const Sdf *sdf, int n, const Particle*, /**/ int *labels);
}
