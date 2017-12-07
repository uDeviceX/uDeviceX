#include <stdio.h>
#include <libconfig.h>

int main(int argc, char **argv) {
    config_t cfg;
    config_init(&cfg);
    config_destroy(&cfg);
    return 0;
}
