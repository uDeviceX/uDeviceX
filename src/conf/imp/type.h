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

enum {
    MAX_LEVEL = 10 /* maximum level of parameters */
};

struct CBuf {
    char c[MAX_LEVEL][FILENAME_MAX];
};
