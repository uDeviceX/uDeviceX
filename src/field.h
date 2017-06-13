namespace field {
void ini_dims(const char *path, int N[3], float extent[3]);
void ini_data(const char *path, const int n, float *grid_data);
void sample(const float rlo[3], const float dr[3], const int nsize[3], const int N[3], const float ampl, const float *grid_data, float *out);
void dump(const int N[], const float extent[], const float* grid_data);
}
