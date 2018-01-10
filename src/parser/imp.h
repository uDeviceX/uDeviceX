struct Config;
struct float3;

// tag::mem[]
void conf_ini(/**/ Config **c);
void conf_fin(/**/ Config *c);
// end::mem[]

// tag::ini[]
void conf_read(int argc, char **argv, /**/ Config *cfg);
// end::ini[]

// tag::lookup[]
void conf_lookup_int(const Config *c, const char *desc, int *a);
void conf_lookup_float(const Config *c, const char *desc, float *a);
void conf_lookup_bool(const Config *c, const char *desc, int *a);
void conf_lookup_string(const Config *c, const char *desc, const char **a);
void conf_lookup_vint(const Config *c, const char *desc, int *n, int a[]);
void conf_lookup_vfloat(const Config *c, const char *desc, int *n, float a[]);
void conf_lookup_float3(const Config *c, const char *desc, float3 *a);
// end::lookup[]

// tag::opt[]
bool conf_opt_int(const Config *c, const char *desc, int *a);
bool conf_opt_float(const Config *c, const char *desc, float *a);
bool conf_opt_bool(const Config *c, const char *desc, int *a);
bool conf_opt_string(const Config *c, const char *desc, const char **a);
bool conf_opt_vint(const Config *c, const char *desc, int *n, int a[]);
bool conf_opt_vfloat(const Config *c, const char *desc, int *n, float a[]);
bool conf_opt_float3(const Config *c, const char *desc, float3 *a);
// end::opt[]
