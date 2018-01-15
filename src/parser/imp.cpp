#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libconfig.h>
#include <vector_types.h>

#include "utils/error.h"
#include "utils/imp.h"
#include "utils/msg.h"

#include "imp.h"

// tag::struct[]
enum {
    EXE, /* from program         */
    ARG, /* from arguments       */
    OPT, /* from additional file */
    DEF, /* from default file    */
    NCFG
};

struct Config {
    config_t c[NCFG];
};
// end::struct[]

void conf_ini(/**/ Config **c) {
    UC(emalloc(sizeof(Config), (void**) c));

    Config *cfg = *c;
    for (int i = 0; i < NCFG; ++i)
        config_init(cfg->c + i);
}

void conf_fin(/**/ Config *c) {
    for (int i = 0; i < NCFG; ++i)
        config_destroy(c->c + i);
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
    msg_print("read config from <%s>", fname);
    if (!config_read_file(c, fname))
        ERR( "%s:%d - %s\n", config_error_file(c),
             config_error_line(c), config_error_text(c));
}

static void read_args(int argc, char **argv, /**/ config_t *c) {
   enum {MAX_CHAR = 100000};
   char *args;

   UC(emalloc(MAX_CHAR * sizeof(char), (void **) &args));

   concatenate(argc, argv, /**/ args);
   if (!config_read_string(c, args)) {
       msg_print("read args: %s", args);
       ERR("%d - %s\n",
           config_error_line(c), config_error_text(c));
   }

   UC(efree(args));
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

    UC(read_file(defname, /**/ &cfg->c[DEF]));

    if (get_opt_file(&argc, &argv, /**/ optname)) {
        UC(read_file(optname, /**/ &cfg->c[OPT]));
    }

    if (argc)
        UC(read_args(argc, argv, /**/ &cfg->c[ARG]));
}

static bool found(int s) {return s == CONFIG_TRUE;}
static bool found(const config_setting_t *s) {return s != NULL;}

static bool lookup_int(const Config *c, const char *desc, int *a) {
    int i, s;
    for (i = 0; i < NCFG; ++i) {
        s = config_lookup_int(c->c + i, desc, /**/ a);
        if ( found(s) ) return true;
    }
    return false;
}

static bool lookup_float(const Config *c, const char *desc, float *a) {
    int i, s;
    double d;

    for (i = 0; i < NCFG; ++i) {
        s = config_lookup_float(c->c + i, desc, /**/ &d);
        *a = d;
        if ( found(s) ) return true;
    }
    return false;
}

static bool lookup_bool(const Config *c, const char *desc, int *a) {
    int i, s;
    for (i = 0; i < NCFG; ++i) {
        s = config_lookup_bool(c->c + i, desc, a);
        if ( found(s) ) return true;
    }
    return false;
}

static bool lookup_string(const Config *c, const char *desc, const char **a) {
    int i, s;
    for (i = 0; i < NCFG; ++i) {
        s = config_lookup_string(c->c + i, desc, a);
        if ( found(s) ) return true;
    }
    return false;
}

static bool lookup_vint(const Config *c, const char *desc, int *n, int a[]) {
    int i, j;
    config_setting_t *s;
    *n = 0;
    for (i = 0; i < NCFG; ++i) {
        s = config_lookup(c->c + i, desc);
        if ( found(s) ) {
            *n = config_setting_length(s);
            for (j = 0; j < *n; ++j)
                a[j] = config_setting_get_int_elem(s, j);
            return true;
        }
    }
    return false;
}

static bool lookup_vfloat(const Config *c, const char *desc, int *n, float a[]) {
    int i, j;
    config_setting_t *s;
    *n = 0;
    for (i = 0; i < NCFG; ++i) {
        s = config_lookup(c->c + i, desc);
        if ( found(s) ) {
            *n = config_setting_length(s);
            for (j = 0; j < *n; ++j)
                a[j] = config_setting_get_float_elem(s, j);
            return true;
        }
    }
    return false;
}

static bool lookup_float3(const Config *c, const char *desc, float3 *a) {
    enum {X, Y, Z};
    int n;
    float f[3];
    bool ret = lookup_vfloat(c, desc, &n, f);
    if (n != 3)
        ERR("fail to read `%s`: float3 must have 3 components, found %d", desc, n);
    a->x = f[X];
    a->y = f[Y];
    a->z = f[Z];
    return ret;
}

void conf_lookup_int(const Config *c, const char *desc, int *a) {
    bool found = lookup_int(c, desc, a);
    if (!found) ERR("Could not find the field <%s>\n", desc);
}

void conf_lookup_float(const Config *c, const char *desc, float *a) {
    bool found = lookup_float(c, desc, a);
    if (!found) ERR("Could not find the field <%s>\n", desc);
}

void conf_lookup_bool(const Config *c, const char *desc, int *a) {
    bool found = lookup_bool(c, desc, a);
    if (!found) ERR("Could not find the field <%s>\n", desc);
}

void conf_lookup_string(const Config *c, const char *desc, const char **a) {
    bool found = lookup_string(c, desc, a);
    if (!found) ERR("Could not find the field <%s>\n", desc);
}

void conf_lookup_vint(const Config *c, const char *desc, int *n, int a[]) {
    bool found = lookup_vint(c, desc, n, a);
    if (!found) ERR("Could not find the field <%s>\n", desc);
}

void conf_lookup_vfloat(const Config *c, const char *desc, int *n, float a[]) {
    bool found = lookup_vfloat(c, desc, n, a);
    if (!found) ERR("Could not find the field <%s>\n", desc);
}

void conf_lookup_float3(const Config *c, const char *desc, float3 *a) {
    bool found = lookup_float3(c, desc, a);
    if (!found) ERR("Could not find the field <%s>\n", desc);
}

bool conf_opt_int(const Config *c, const char *desc, int *a) {
    return lookup_int(c, desc, a);
}

bool conf_opt_float(const Config *c, const char *desc, float *a)  {
    return lookup_float(c, desc, a);
}

bool conf_opt_bool(const Config *c, const char *desc, int *a)  {
    return lookup_bool(c, desc, a);
}

bool conf_opt_string(const Config *c, const char *desc, const char **a)  {
    return lookup_string(c, desc, a);
}

bool conf_opt_vint(const Config *c, const char *desc, int *n, int a[]) {
    return lookup_vint(c, desc, n, a);
}

bool conf_opt_vfloat(const Config *c, const char *desc, int *n, float a[]) {
    return lookup_vfloat(c, desc, n, a);
}

bool conf_opt_float3(const Config *c, const char *desc, float3 *a) {
    return lookup_float3(c, desc, a);
}



void conf_set_int(int n, const char *desc[], int a, Config *cfg) {
    config_t *c;
    config_setting_t *root, *group, *s;
    c = &cfg->c[EXE];
    
    root  = config_root_setting(c);
    group = root;

    for (int i = 0; i < n - 1; ++i)
        group = config_setting_add(group, desc[i], CONFIG_TYPE_GROUP);
    
    s = config_setting_add(group, desc[n-1], CONFIG_TYPE_INT);
    config_setting_set_int(s, a);
    config_write(c, stderr);
}
