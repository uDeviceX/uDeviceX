#pragma once

enum
{
    XSIZE_SUBDOMAIN = 160,
    YSIZE_SUBDOMAIN = 40,
    ZSIZE_SUBDOMAIN = 8,
    XMARGIN_WALL = 6,
    YMARGIN_WALL = 6,
    ZMARGIN_WALL = 6,
};

const int numberdensity = 3;
const float dt = 0.005;
const float kBT = 0.01;
const float gammadpd = 80;
const float sigma = sqrt(2 * gammadpd * kBT);
const float sigmaf = sigma / sqrt(dt);
const float aij = 2;
const float hydrostatic_a = 0.0003125;
