struct Coords;
struct Tform;

void sdf_field_ini_dims(const char *path, /**/ int N[3],    float ext[3]);
void sdf_field_ini_data(const char *path, int n, /**/ float *D);
void sdf_field_sample(Tform*, const int N0[3], const float *D0, const int N1[3],
            /**/ float *D1);
void sdf_field_scale(const int N[3], float s, /**/ float *D);
void sdf_field_dump(const Coords*, MPI_Comm cart, const int N[3], const float *D);
