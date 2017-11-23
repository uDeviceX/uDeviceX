namespace restrain {

/* return statistics of restrain: number of particle and center of
   mass velocity */
void stat(/**/ int *n, float *v);

namespace color {
void vel(const int *cc, int color, int n, /**/ Particle *pp);
}

namespace grey {
void vel(int n, /**/ Particle *pp);
}

}
