struct Tform;

// tag::interface[]
void tform_ini(Tform**, float a0[3], float a1[3], float b0[3], float b1[3]);
void tform_fin(Tform*);

void tform_0to1(Tform*, float a0[3], /**/ float a1[3]);
// end::interface[]
