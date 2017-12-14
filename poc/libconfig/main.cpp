#include <stdio.h>
#include <string.h>
#include <libconfig.h>

struct Config {
    config_t args;
    config_t file;
};

void conf_ini(/**/ Config *c);
void conf_read_args(int argc, char **argv, /**/ Config *c);
void conf_read_file(const char *fname, /**/ Config *c);
void conf_lookup_int(const Config *c, const char *desc, int *a);
void conf_destroy(/**/ Config *c);

int main(int argc, char **argv) {
    Config c;

    conf_ini(&c);
    conf_read_args(argc-1, argv + 1, &c);
    conf_read_file("default.cfg", &c);
    
    int a, b;
    conf_lookup_int(&c, "a", &a);
    conf_lookup_int(&c, "b", &b);

    printf("%d\n%d\n", a, b);
    
    conf_destroy(&c);
    return 0;
}

void conf_ini(/**/ Config *c) {
    config_init(&c->args);
    config_init(&c->file);
}

void conf_read_args(int argc, char **argv, /**/ Config *c) {
    enum {MAX_CHAR=100000};
    char *args, *a;
    int i;    
    args = new char[MAX_CHAR];
    args[0] = '\0';

    for(i = 0; i < argc; ++i) {
        a = argv[i];
        strcat(args, a);
        strcat(args, " ");
    }

    config_read_string(&c->args, args);
    
    delete[] args;    
}

void conf_read_file(const char *fname, /**/ Config *c) {
    config_read_file(&c->file, fname);
}

static bool found(int s) {return s == CONFIG_TRUE;}

void conf_lookup_int(const Config *c, const char *desc, int *a) {
    int s;
    s = config_lookup_int(&c->args, desc, a);
    if ( found(s) ) return;
    s = config_lookup_int(&c->file, desc, a);
}

void conf_destroy(/**/ Config *c) {
    config_destroy(&c->args);
    config_destroy(&c->file);
}

/*

# TEST: default.t0
make -s
./main > res.out.txt

# TEST: b.t0
make -s
./main b=4 > res.out.txt

# TEST: ab.t0
make -s
./main a=2 b=4 > res.out.txt

*/
