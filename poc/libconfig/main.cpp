#include <stdio.h>
#include <libconfig.h>

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
