namespace os { /* [o]perating [s]ystems commands */
void mkdir(const char* path);
long time();
void srand(long int seedval);
double drand();

/* get an environment variable
   *_d: with default */
void env2float  (const char*,            float *px);
void env2float_d(const char*, float def, float *px);
}
