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
    EXE, /* from program setters */
    ARG, /* from arguments       */
    OPT, /* from additional file */
    DEF, /* from default file    */
    NCFG
};

enum {INI = 123}; /* status */
struct Config {
    int status;
    config_t c[NCFG];
};
// end::struct[]

enum {
    OK,
    NOTFOUND,
    WTYPE
};

void conf_ini(/**/ Config **pq) {
    Config *q;
    EMALLOC(1, &q);
    for (int i = 0; i < NCFG; ++i)
        config_init(q->c + i);
    q->status = INI;
    *pq = q;
}

void conf_fin(/**/ Config *q) {
    int i;
    if (q->status != INI) ERR("wrong conf_fin call");
    for (i = 0; i < NCFG; ++i)
        config_destroy(&q->c[i]);
    EFREE(q);
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
    msg_print("read '%s'", fname);
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

static void set_include_dir(const char *path, Config *cfg) {
    for (int i = 0; i < NCFG; ++i)
        config_set_include_dir(&cfg->c[i], path);
}

void conf_read(int argc, char **argv, /**/ Config *cfg) {
    char *home, defname[1024] = {0}, optname[1024], definclude[1024] = {0};
    home = getenv("HOME");

    // default include dir
    strcpy(definclude, home);
    strcat(definclude, "/.udx/");
    set_include_dir(definclude, cfg);
    
    // default parameters file
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

static int lookup_setting(const Config *c, const char *desc, config_setting_t **s) {
    int i;
    for (i = 0; i < NCFG; ++i) {
        *s = config_lookup(c->c + i, desc);
        if ( found(*s) ) return OK;
    }
    return NOTFOUND;
}

static int lookup_int(const Config *c, const char *desc, int *a) {
    config_setting_t *s;
    int status;
    status = lookup_setting(c, desc, /**/ &s);
    if (status != OK) return status;
    if (config_setting_type(s) != CONFIG_TYPE_INT) return WTYPE;
    *a = config_setting_get_int(s);
    return status;
}

static bool lookup_float(const Config *c, const char *desc, float *a) {
    config_setting_t *s;
    int status;
    status = lookup_setting(c, desc, /**/ &s);
    if (status != OK) return status;
    if (config_setting_type(s) != CONFIG_TYPE_FLOAT) return WTYPE;
    *a = config_setting_get_float(s);
    return status;
}

static bool lookup_bool(const Config *c, const char *desc, int *a) {
    config_setting_t *s;
    int status;
    status = lookup_setting(c, desc, /**/ &s);
    if (status != OK) return status;
    if (config_setting_type(s) != CONFIG_TYPE_BOOL) return WTYPE;
    *a = config_setting_get_bool(s);
    return status;
}

static bool lookup_string(const Config *c, const char *desc, const char **a) {
    config_setting_t *s;
    int status;
    status = lookup_setting(c, desc, /**/ &s);
    if (status != OK) return status;
    if (config_setting_type(s) != CONFIG_TYPE_STRING) return WTYPE;
    *a = config_setting_get_string(s);
    return status;
}

static bool lookup_vint(const Config *c, const char *desc, int *n, int a[]) {
    config_setting_t *s;
    int j, status;
    status = lookup_setting(c, desc, /**/ &s);
    if (status != OK) return status;
    if (config_setting_type(s) != CONFIG_TYPE_ARRAY) return WTYPE;
    *n = config_setting_length(s);
    for (j = 0; j < *n; ++j)
        a[j] = config_setting_get_int_elem(s, j);
    return status;
}

static bool lookup_int3(const Config *c, const char *desc, int3 *a) {
    enum {X, Y, Z};
    int status, n = 0, f[3];
    status = lookup_vint(c, desc, &n, f);
    if (status != OK) return status;
    if (n != 3)
        ERR("fail to read `%s`: int3 must have 3 components, found %d", desc, n);
    a->x = f[X];
    a->y = f[Y];
    a->z = f[Z];
    return status;
}

static bool lookup_vfloat(const Config *c, const char *desc, int *n, float a[]) {
    config_setting_t *s;
    int j, status;
    status = lookup_setting(c, desc, /**/ &s);
    if (status != OK) return status;
    if (config_setting_type(s) != CONFIG_TYPE_ARRAY) return WTYPE;
    *n = config_setting_length(s);
    for (j = 0; j < *n; ++j)
        a[j] = config_setting_get_float_elem(s, j);
    return status;
}

static bool lookup_float3(const Config *c, const char *desc, float3 *a) {
    enum {X, Y, Z};
    int n = 0, status;
    float f[3];
    status = lookup_vfloat(c, desc, &n, f);
    if (status != OK) return status;
    if (n != 3)
        ERR("fail to read `%s`: float3 must have 3 components, found %d", desc, n);
    a->x = f[X];
    a->y = f[Y];
    a->z = f[Z];
    return status;
}

void conf_lookup_int(const Config *c, const char *desc, int *a) {
    int status = lookup_int(c, desc, a);
    if (OK != status) ERR("Could not find int <%s>\n", desc);
}

void conf_lookup_float(const Config *c, const char *desc, float *a) {
    int status = lookup_float(c, desc, a);
    if (OK != status) ERR("Could not find float  <%s>\n", desc);
}

void conf_lookup_bool(const Config *c, const char *desc, int *a) {
    int status = lookup_bool(c, desc, a);
    if (OK != status) ERR("Could not find bool <%s>\n", desc);
}

void conf_lookup_string(const Config *c, const char *desc, const char **a) {
    int status = lookup_string(c, desc, a);
    if (OK != status) ERR("Could not find string <%s>\n", desc);
}

void conf_lookup_vint(const Config *c, const char *desc, int *n, int a[]) {
    int status = lookup_vint(c, desc, n, a);
    if (OK != status) ERR("Could not find vint <%s>\n", desc);
}

void conf_lookup_int3(const Config *c, const char *desc, int3 *a) {
    int status = lookup_int3(c, desc, a);
    if (OK != status) ERR("Could not find int3 <%s>\n", desc);
}

void conf_lookup_vfloat(const Config *c, const char *desc, int *n, float a[]) {
    int status = lookup_vfloat(c, desc, n, a);
    if (OK != status) ERR("Could not find vfloat <%s>\n", desc);
}

void conf_lookup_float3(const Config *c, const char *desc, float3 *a) {
    int status = lookup_float3(c, desc, a);
    if (OK != status) ERR("Could not find float3 <%s>\n", desc);
}

bool conf_opt_int(const Config *c, const char *desc, int *a) {
    return OK == lookup_int(c, desc, a);
}

bool conf_opt_float(const Config *c, const char *desc, float *a)  {
    return OK == lookup_float(c, desc, a);
}

bool conf_opt_bool(const Config *c, const char *desc, int *a)  {
    return OK == lookup_bool(c, desc, a);
}

bool conf_opt_string(const Config *c, const char *desc, const char **a)  {
    return OK == lookup_string(c, desc, a);
}

bool conf_opt_vint(const Config *c, const char *desc, int *n, int a[]) {
    return OK == lookup_vint(c, desc, n, a);
}

bool conf_opt_int3(const Config *c, const char *desc, int3 *a) {
    return OK == lookup_int3(c, desc, a);
}

bool conf_opt_vfloat(const Config *c, const char *desc, int *n, float a[]) {
    return OK == lookup_vfloat(c, desc, n, a);
}

bool conf_opt_float3(const Config *c, const char *desc, float3 *a) {
    return OK == lookup_float3(c, desc, a);
}

static config_setting_t* subsetting(const char *desc, int type, config_setting_t *group) {
    config_setting_t *sub;
    sub = config_setting_lookup(group, desc);
    if (NULL == sub)
        sub = config_setting_add(group, desc, type);
    return sub;
}

static config_setting_t* get_subgroup_setting(int n, const char *desc[], config_t *c) {
    config_setting_t *group;
    group = config_root_setting(c);
    
    for (int i = 0; i < n; ++i)
        group = subsetting(desc[i], CONFIG_TYPE_GROUP, group);
    return group;    
}

void conf_set_int(int n, const char *desc[], int a, Config *cfg) {
    config_t *c;
    config_setting_t *group, *setting;
    int status;
    c = &cfg->c[EXE];
    
    group = get_subgroup_setting(n-1, desc, /**/ c);
    setting = subsetting(desc[n-1], CONFIG_TYPE_INT, /**/ group);
    
    status = config_setting_set_int(setting, a);
    if (CONFIG_TRUE != status)
        ERR("could not set <%s>", desc[n-1]);
}

void conf_set_vint(int n, const char *desc[], int nelem, const int a[], Config *cfg) {
    config_t *c;
    config_setting_t *group, *setting, *status;
    int i;
    enum {APPEND = -1};
    c = &cfg->c[EXE];
        
    group = get_subgroup_setting(n-1, desc, /**/ c);
    setting = subsetting(desc[n-1], CONFIG_TYPE_ARRAY, /**/ group);

    for (i = 0; i < nelem; ++i) {
        status = config_setting_set_int_elem(setting, APPEND, a[i]);
        if (NULL == status)
            ERR("could not set element %d/%d of <%s>", i, nelem, desc[n-1]);
    }
}

void conf_set_int3(int n, const char *desc[], int3 a, Config *cfg) {
    const int a3[] = {a.x, a.y, a.z};
    UC(conf_set_vint(n, desc, 3, a3, cfg));
}

void conf_set_float(int n, const char *desc[], float a, Config *cfg) {
    config_t *c;
    config_setting_t *group, *setting;
    int status;
    c = &cfg->c[EXE];
    
    group = get_subgroup_setting(n-1, desc, /**/ c);
    setting = subsetting(desc[n-1], CONFIG_TYPE_FLOAT, /**/ group);
    
    status = config_setting_set_float(setting, a);
    if (CONFIG_TRUE != status)
        ERR("could not set <%s>", desc[n-1]);
}

void conf_set_vfloat(int n, const char *desc[], int nelem, const float a[], Config *cfg) {
    config_t *c;
    config_setting_t *group, *setting, *status;
    int i;
    enum {APPEND = -1};
    c = &cfg->c[EXE];
        
    group = get_subgroup_setting(n-1, desc, /**/ c);
    setting = subsetting(desc[n-1], CONFIG_TYPE_ARRAY, /**/ group);

    for (i = 0; i < nelem; ++i) {
        status = config_setting_set_float_elem(setting, APPEND, a[i]);
        if (NULL == status)
            ERR("could not set element %d/%d of <%s>", i, nelem, desc[n-1]);
    }
}

void conf_set_float3(int n, const char *desc[], float3 a, Config *cfg) {
    const float a3[] = {a.x, a.y, a.z};
    UC(conf_set_vfloat(n, desc, 3, a3, cfg));
}

void conf_set_bool(int n, const char *desc[], int a, Config *cfg) {
    config_t *c;
    config_setting_t *group, *setting;
    int status;
    c = &cfg->c[EXE];
    
    group = get_subgroup_setting(n-1, desc, /**/ c);
    setting = subsetting(desc[n-1], CONFIG_TYPE_BOOL, /**/ group);
    
    status = config_setting_set_bool(setting, a);
    if (CONFIG_TRUE != status)
        ERR("could not set <%s>", desc[n-1]);
}

void conf_set_string(int n, const char *desc[], const char *a, Config *cfg) {
    config_t *c;
    config_setting_t *group, *setting;
    int status;
    c = &cfg->c[EXE];
    
    group = get_subgroup_setting(n-1, desc, /**/ c);
    setting = subsetting(desc[n-1], CONFIG_TYPE_STRING, /**/ group);
    
    status = config_setting_set_string(setting, a);
    if (CONFIG_TRUE != status)
        ERR("could not set <%s>", desc[n-1]);
}

void conf_write_exe(const Config *cfg, FILE *stream) {
    config_write(&cfg->c[EXE], stream);
}
