static __device__ void tform_convert_dev(Tform *t, float a0[3], /**/ float a1[3]) {
    enum {X, Y, Z};
    float *o, *s;
    o = t->o; s = t->s;
    a1[X] = s[X]*a0[X] + o[X];
    a1[Y] = s[Y]*a0[Y] + o[Y];
    a1[Z] = s[Z]*a0[Z] + o[Z];
}
