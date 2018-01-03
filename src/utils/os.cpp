#include <stdio.h>
#include <stdlib.h>

#include <sys/stat.h>
#include <errno.h>
#include <time.h>

#include "msg.h"
#include "utils/error.h"

#include "os.h"

namespace os {
void mkdir(const char *path) {
    mode_t mode;
    int rc, ok;
    mode = S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH;
    rc = ::mkdir(path, mode);
    ok = (rc == 0 || errno == EEXIST);
    if (!ok) {
        msg_print("os::mkdir: cannot create directory ‘%s’", path);
        ERR("errno: %d\n", errno);
        exit(1);
    }
}

long time() { return ::time(NULL); }
void   srand(long int seedval) { srand48(seedval); }
double drand() { return drand48(); }

enum {OK, NOT_SET, NOT_FLOAT};
static int env2float0(const char *key, /**/ const char **pval, float *px) {
    if ((*pval = getenv(key)) == NULL)
        return NOT_SET;
    if (sscanf(*pval, "%f", px) != 1)
        return NOT_FLOAT;
    return OK;
}

void env2float(const char *key, float *px) {
    int status;
    const char *val;
    status = env2float0(key, /**/ &val, px);
    if (status == OK)
        msg_print("env %s = %g", key, *px);
    else if (status == NOT_SET)
        ERR("env. variable is not set: `%s`", key, val);
    else if (status == NOT_FLOAT)
        ERR("env. variable `%s = %s` is not float", key, val);
    else
        ERR("unknown status");
}

void env2float_d(const char *key, float def, /**/ float *px) { /* with default */
    int status;
    const char *val;
    status = env2float0(key, /**/ &val, px);
    if (status == OK)
        msg_print("env %s = %g", key, *px);
    else if (status == NOT_SET) {
        *px = def;
        msg_print("env %s = %g (default)", key, *px);
    } else if  (status == NOT_FLOAT)
        ERR("env. variable `%s = %s` is not float", key, val);
    else
        ERR("unknown status");
}

} /* namespace */
