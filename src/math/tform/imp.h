struct Tform;
struct Tform_v;

// tag::interface[]
void tform_ini(Tform**);
void tform_fin(Tform*);

void tform_vector(float a0[3], float a1[3],   float b0[3], float b1[3], /**/ Tform*);
void tform_chain(Tform*, Tform*, /**/ Tform*);
void tform_convert(Tform*, float a0[3], /**/ float a1[3]);

void tform_log(Tform*);

void tform_2view(Tform*, Tform_v*);
// end::interface[]
