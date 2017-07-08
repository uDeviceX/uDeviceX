namespace field {
void ini_dims(const char *path, int N[3], float extent[3]);
void ini_data(const char *path, const int n, float *grid_data);
void sample(const float org[3], const float spa[3], const int N1[3], const int N0[3], const float *D0, float *D1);
void scale(int N[3], float s, float *D);
void dump(const int N[], const float* grid_data);
}
