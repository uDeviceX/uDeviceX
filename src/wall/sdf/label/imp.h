enum {
    LABEL_BULK,  /* remains in bulk */
    LABEL_WALL,  /* becomes wall particle */
    LABEL_DEEP   /* deep inside the wall */
};
void wall_label_dev(const Sdf *sdf, int n, const Particle*, /**/ int *labels);
void wall_label_hst(const Sdf *sdf, int n, const Particle*, /**/ int *labels);
