struct Coords;
struct Tform;
struct Field;

void field_ini(const char *path, /**/ Field**);
void field_sample(const Field*, Tform*, const int N[3], /**/ Field**);

void field_size(const Field*, /**/ int N[3]);
void field_extend(const Field*, /**/ float ext[3]);
void field_data(const Field*, /**/ float**);

void field_dump(const Field*, const Coords*, MPI_Comm cart);
void field_scale(Field*, float scale);
void field_fin(Field*);
