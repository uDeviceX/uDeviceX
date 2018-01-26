#include <stdio.h>

#include "utils/error.h"
#include "utils/imp.h"
#include "parser/imp.h"

#include "imp.h"

void inter_color_set_conf(const Config *cfg, GenColor *gc) {
    const char *kind;
    UC(conf_lookup_string(cfg, "color.kind", &kind));

    if      (same_str(kind, "uniform")) {
        inter_color_set_uniform(gc);
    }
    else if (same_str(kind, "drop")) {
        float R;
        UC(conf_lookup_float(cfg, "color.R", &R));
        inter_color_set_drop(R, gc);
    }
}
