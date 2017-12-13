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
void conf_destroy(/**/ Config *c);

int main(int argc, char **argv) {
    config_t cfg;
    int i;
    config_init(&cfg);

    for (i = 1; i < argc; ++i) {
        printf("parsing %s\n", argv[i]);
        config_read_string(&cfg, argv[i]);
    }

    int a, b;
    config_lookup_int(&cfg, "a", &a);
    config_lookup_int(&cfg, "b", &b);

    printf("a = %d, b = %d\n", a, b);
    
    config_destroy(&cfg);
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

void conf_destroy(/**/ Config *c) {
    config_destroy(&c->args);
    config_destroy(&c->file);
}
