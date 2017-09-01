# "local" Cloud

`Pa`, `Index` are abstract particle and index.

Cloud is a family of abstract object with API.

    Pa = pa(Cloud, Index)
    r  = r(Cloud, Index)

The `__global__` function recives one or two Cloud objects. For
example,

    fun(CloudA, CloudB, [...])

The function uses only API of clouds.

