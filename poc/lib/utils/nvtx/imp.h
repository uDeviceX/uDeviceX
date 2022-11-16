void nvtx_push(const char *name);
void nvtx_pop();

#define NVTX_PUSH(name) nvtx_push(name)
#define NVTX_POP() nvtx_pop()
