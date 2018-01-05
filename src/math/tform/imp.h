struct Tform;
struct Tform_v;

// tag::interface[]
void tform_ini(Tform**); // <I>
void tform_fin(Tform*);

void tform_vector(const float a0[3], const float a1[3],   const float b0[3], const float b1[3], /**/ Tform*); // <C>
void tform_chain(Tform*, Tform*, /**/ Tform*);
void tform_grid2grid(const float f_lo[3], const float f_hi[3], const int f_n[3],
                     const float t_lo[3], const float t_hi[3], const int t_n[3], /**/
                     Tform*);
void tform_to_grid(const float a0[3], const float b0[3], const int n[3], /**/ Tform*);
void tform_from_grid(const float a0[3], const float b0[3], const int n[3], /**/ Tform*);

void tform_convert(Tform*, const float a0[3], /**/ float a1[3]); // <U>

void tform_to_view(Tform*, /**/ Tform_v*); // <V>

void tform_log(Tform*); // <L>
void tform_dump(Tform*, FILE*);
// end::interface[]
