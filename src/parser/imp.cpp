#include <stdio.h>
#include <string.h>
#include <libconfig.h>

#include "utils/error.h"
#include "utils/imp.h"

#include "imp.h"

// tag::struct[]
struct Config {
    config_t args; /* from arguments       */
    config_t file; /* from additional file */
    config_t def;  /* from default file    */
};
// end::struct[]

void conf_ini(/**/ Config **c) {
    UC(emalloc(sizeof(Config), (void**) c));

    Config *cfg = *c;
    config_init(&cfg->args);
    config_init(&cfg->file);
    config_init(&cfg->def);
}

void conf_destroy(/**/ Config *c) {
    config_destroy(&c->args);
    config_destroy(&c->file);
    config_destroy(&c->def);
    UC(efree(c));
}

static void concatenate(int n, char **ss, /**/ char *a) {
    char *s;
    a[0] = '\0';

    for(int i = 0; i < n; ++i) {
        s = ss[i];
        strcat(a, s);
        strcat(a, " ");
    }
}

void conf_read_file(const char *fname, /**/ Config *cfg) {
    config_t *c = &cfg->file;

    if (!config_read_file(c, fname))
        ERR( "%s:%d - %s\n", config_error_file(c),
             config_error_line(c), config_error_text(c));
}

void conf_read_args(int argc, char **argv, /**/ Config *cfg) {
    enum {MAX_CHAR = 100000};
    char *args;
    config_t *c = &cfg->args;
    
    UC(emalloc(MAX_CHAR * sizeof(char), (void **) &args));

    concatenate(argc, argv, /**/ args);

    if (!config_read_string(c, args))
        ERR( "arguments: %d - %s\n",
             config_error_line(c), config_error_text(c));
    
    delete[] args;    
}

static bool found(int s) {return s == CONFIG_TRUE;}

void conf_lookup_int(const Config *c, const char *desc, int *a) {
    int s;
    s = config_lookup_int(&c->args, desc, a);
    if ( found(s) ) return;
    s = config_lookup_int(&c->file, desc, a);
    if ( found(s) ) return;
    s = config_lookup_int(&c->def, desc, a);
    if ( found(s) ) return;
}

void conf_lookup_float(const Config *c, const char *desc, float *a) {
    int s;
    double d;
    s = config_lookup_float(&c->args, desc, &d);
    *a = d;
    if ( found(s) ) return;

    s = config_lookup_float(&c->file, desc, &d);
    *a = d;
    if ( found(s) ) return;

    s = config_lookup_float(&c->def, desc, &d);
    *a = d;
    if ( found(s) ) return;
}

void conf_lookup_bool(const Config *c, const char *desc, int *a) {
    int s;
    s = config_lookup_bool(&c->args, desc, a);
    if ( found(s) ) return;
    s = config_lookup_bool(&c->file, desc, a);
    if ( found(s) ) return;
    s = config_lookup_bool(&c->def, desc, a);
    if ( found(s) ) return;
}

void conf_lookup_string(const Config *c, const char *desc, const char **a) {
    int s;
    s = config_lookup_string(&c->args, desc, a);
    if ( found(s) ) return;
    s = config_lookup_string(&c->file, desc, a);
    if ( found(s) ) return;
    s = config_lookup_string(&c->def, desc, a);
    if ( found(s) ) return;
}
