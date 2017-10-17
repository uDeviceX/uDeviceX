enum {
    XOFFSET = XS / 2,
    YOFFSET = YS / 2,
    ZOFFSET = ZS / 2
};

namespace c { /* common */
texture<int, cudaTextureType1D> starts, id;
__constant__ const float2 *PP[MAX_OBJ_TYPES];
__constant__ float *FF[MAX_OBJ_TYPES];
}
