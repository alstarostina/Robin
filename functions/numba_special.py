"""
We wrap some of the special functions in scipy which have cython support for numba.

For this we will use that numba can use arbitrary cython functions if we define them properly. This documented in detail in [1].
The main question we have to pose: How can we guess the weird names like __pyx_fuse_1kv or __pyx_fuse_1gamma below?
We will exemplary show this for the Bessel function of the second kind, i.e. ```scipy.special.cython_special.kv```.
We will basically query the ```__pyx_capi___``` attribute which is an internal "table of content" for cython modules [2].
We start an ipython in a terminal and type
```
>>> import scipy                                                                                                                                                     
>>> import scipy.special                                                                                                                                             
>>> import scipy.special.cython_special                                                                                                                              
```
Then look at the table of contents with
```
>>> all_functions = list(scipy.special.cython_special.__pyx_capi__.items())                                                                                          
```
Since the module defines many functions, we filter for all functions with the letters kv in their name, i.e.,
```
>>> [x for x in all_functions if x[0].find('kv') != -1]                                                                                                              
[('__pyx_fuse_0kv',
  <capsule object "__pyx_t_double_complex (double, __pyx_t_double_complex, int __pyx_skip_dispatch)" at 0x7f7148cbf3f0>),
 ('__pyx_fuse_1kv',
  <capsule object "double (double, double, int __pyx_skip_dispatch)" at 0x7f7148cbf420>),
 ('__pyx_fuse_0kve',
  <capsule object "__pyx_t_double_complex (double, __pyx_t_double_complex, int __pyx_skip_dispatch)" at 0x7f7148cbf450>),
 ('__pyx_fuse_1kve',
  <capsule object "double (double, double, int __pyx_skip_dispatch)" at 0x7f7148cbf480>)]
```
We get a list of tuples consisting of the function name (__pyx_fuse_...) and the type signature (<capsule object ....).
Clearly ```__pyx_fuse_0kve``` and ```__pyx_fuse_1kve``` correspond to the function ```scipy.special.kve``` which is not what we want.
```__pyx_fuse_0kv``` takes complex values as input and output parameter, also not what we want.
So, ```__pyx_fuse_1kv``` seems to be the right one, acting on doubles!

[1] https://numba.pydata.org/numba-doc/dev/extending/high-level.html#importing-cython-functions
[2] https://www.bountysource.com/issues/78681198-questions-about-__pyx_capi__-use-stability
"""

import ctypes
from numba.extending import get_cython_function_address


addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1kv")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
kv = functype(addr)

addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1gamma")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
gamma = functype(addr)


# small test to make sure we have the correct thing
if __name__ == '__main__':
    import numpy as np
    import scipy.special as ss 

    for x in [0.2, 0.5, 1, 1.5, 2]:
        np.testing.assert_allclose(ss.gamma(x), gamma(x))

    for n in [0, 1, 2, 3]:
        for x in [0, 0.2, 0.5, 1, 1.5, 2]:
            np.testing.assert_allclose(ss.kv(n, x), kv(n, x))
