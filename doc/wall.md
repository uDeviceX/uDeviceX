# int
    ``` cpp
    struct Quants {
        float4 *pp;
        int n;
        Logistic::KISS *rnd;
        Clist *cells;
        cudaTextureObject_t texstart;
        cudaTextureObject_t texpp;
    }
    ```

# wall functions called by sim::
    ```cpp
    void alloc_quants(Quants *q);
    void free_quants(Quants *q);
    int create(int n, Particle* pp, Quants *q);
    void interactions(const Quants q, const int type, const Particle *pp, const int n, Force *ff);
    
    ```
