struct Config {
    config_t args;
    config_t file;
};

void conf_ini(/**/ Config *c);
void conf_destroy(/**/ Config *c);

void conf_read_args(int argc, char **argv, /**/ Config *c);
void conf_read_file(const char *fname, /**/ Config *c);

void conf_lookup_int(const Config *c, const char *desc, int *a);
void conf_lookup_float(const Config *c, const char *desc, float *a);
void conf_lookup_bool(const Config *c, const char *desc, int *a);
void conf_lookup_string(const Config *c, const char *desc, const char **a);
