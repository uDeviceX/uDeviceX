static __device__ void tform_convert_dev(Tform_v *t, const float a0[3], /**/ float a1[3]) {
    enum {X, Y, Z};
    float *o, *s;
    o = t->o; s = t->s;
    a1[X] = s[X]*a0[X] + o[X];
    a1[Y] = s[Y]*a0[Y] + o[Y];
    a1[Z] = s[Z]*a0[Z] + o[Z];
}

static __device__ void tform_spacing_dev(Tform_v *t, /**/ float s[3]) {
    enum {X, Y, Z};
    s[X] = t->s[X];
    s[Y] = t->s[Y];
    s[Z] = t->s[Z];
}
