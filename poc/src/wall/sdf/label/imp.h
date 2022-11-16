enum {
    // tag::enum[]
    LABEL_BULK,  /* remains in bulk */
    LABEL_WALL,  /* becomes wall particle */
    LABEL_DEEP   /* deep inside the wall */
    // end::enum[]
};

// tag::int[]
void wall_label_dev(const Sdf *sdf, int n, const Particle*, /**/ int *labels); // <1>
void wall_label_hst(const Sdf *sdf, int n, const Particle*, /**/ int *labels); // <2>
// end::int[]
