struct Coords;
struct Tform;
struct Field;

// tag::mem[]
void field_ini(const char *path, /**/ Field**); // <1>
void field_fin(Field*);                         // <2>
// end::mem[]

// tag::int[]
void field_sample(const Field*, const Tform*, const int N[3], /**/ Field**); // <1>
// end::int[]

// tag::get[]
void field_size(const Field*, /**/ int N[3]);       // <1>
void field_extend(const Field*, /**/ float ext[3]); // <2>
void field_data(const Field*, /**/ float**);        // <3>
// end::get[]

// tag::int[]
void field_dump(const Field*, const Coords*, MPI_Comm cart); // <2>
void field_scale(Field*, float scale); // <3>
// end::int[]
