#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libconfig.h>

#include "utils/error.h"
#include "utils/imp.h"
#include "msg.h"

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

static void read_file(const char *fname, /**/ config_t *c) {
    MSG("read config from <%s>", fname);
    if (!config_read_file(c, fname))
        ERR( "%s:%d - %s\n", config_error_file(c),
             config_error_line(c), config_error_text(c));
}

static void read_args(int argc, char **argv, /**/ config_t *c) {
   enum {MAX_CHAR = 100000};
    char *args;
    
    UC(emalloc(MAX_CHAR * sizeof(char), (void **) &args));

    concatenate(argc, argv, /**/ args);

    if (!config_read_string(c, args))
        ERR( "arguments: %d - %s\n",
             config_error_line(c), config_error_text(c));
    
    delete[] args;
}

static void shift(int *c, char ***v) {
    (*c) --;
    (*v) ++;
}

static int get_opt_file(int *argc, char ***argv, /**/ char fname[]) {
    char *lastpnt, *a;
    int differ;

    if (*argc) {
        a = (*argv)[0];
        lastpnt = strrchr(a, '.');
        
        if (lastpnt != NULL) {
            differ = strcmp(lastpnt, ".cfg");
            if (differ) return 0;
            strcpy(fname, a);
            shift(argc, argv);
            return 1;
        }
    }
    return 0;
}

void conf_read(int argc, char **argv, /**/ Config *cfg) {
    char *home, defname[1024] = {0}, optname[1024];
    home = getenv("HOME");

    strcpy(defname, home);
    strcat(defname, "/.udx/default.cfg");

    UC(read_file(defname, /**/ &cfg->def)); 
    
    if (get_opt_file(&argc, &argv, /**/ optname)) {
        UC(read_file(optname, /**/ &cfg->file));
    }

    if (argc)
        UC(read_args(argc, argv, /**/ &cfg->args));
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
