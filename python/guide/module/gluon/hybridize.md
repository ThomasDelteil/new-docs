# A Hybrid of Imperative and Symbolic Programming

Hybridization is the technique used by MXNet Gluon in order to combine the imperative and symbolic programming paradigms. It is **strongly recommended** to read more about it and find examples on how to write hybridizable networks by following the pre-requisite tutorial.
The second part covers the main differences and limitations between fully imperative networks and hybridizable ones.

## Pre-requisite

Understand the difference between [Imperative and Symbolic Programming with MXNet](https://d2l.ai/chapter_computational-performance/hybridize.html) and how to create hybridized graphs with [`HybridBlock`](http://beta.mxnet.io/api/gluon/mxnet.gluon.nn.HybridBlock.html) and [`.hybridize()`](http://beta.mxnet.io/api/gluon/_autogen/mxnet.gluon.nn.HybridBlock.hybridize.html#mxnet.gluon.nn.HybridBlock.hybridize). 

## Key differences and limitations of hybridization

The difference between a purely imperative `Block` and hybridizable `HybridBlock` can superficially appear to be simply the injection of the `F` function space (resolving to [`mx.nd`](http://beta.mxnet.io/api/ndarray/index.html) or [`mx.sym`](http://beta.mxnet.io/api/symbol/index.html)) in the forward function that is renamed from `forward` to `hybrid_forward`. However there are some limitations that apply when using hybrid blocks. In the following section we will review the main differences, giving example of code snippets that generate errors when such blocks get hybridized.

### Indexing

When trying to access specific elements in a tensor like this:

```python
def hybrid_forward(self, F, x):
    return x[0,0]
```

Would generate the following error:

`TypeError: Symbol only support integer index to fetch i-th output`

There are however several operators that can help you with array manipulations like: [`F.split`](http://beta.mxnet.io/api/ndarray/_autogen/mxnet.ndarray.split.html#mxnet.ndarray.split), [`F.slice`](http://beta.mxnet.io/api/ndarray/_autogen/mxnet.ndarray.slice.html#mxnet.ndarray.slice), [`F.take`](http://beta.mxnet.io/api/ndarray/_autogen/mxnet.ndarray.take.html),[`F.pick`](http://beta.mxnet.io/api/ndarray/_autogen/mxnet.ndarray.pick.html), [`F.where`](http://beta.mxnet.io/api/ndarray/_autogen/mxnet.ndarray.where.html), [`F.reshape`](http://beta.mxnet.io/api/ndarray/_autogen/mxnet.ndarray.reshape.html) or [`F.reshape_like`](http://beta.mxnet.io/api/ndarray/_autogen/mxnet.ndarray.reshape_like.html).

### Data Type

Sometimes one can be tempted to use conditionnal logic on the type of the input tensors however the following block:

```python
def hybrid_forward(self, F, x):
    if x.dtype =='float16':
        return x
    return x*2
```

Would generate a `AttributeError: 'Symbol' object has no attribute 'dtype'`

You cannot use the `dtype` of the symbol at runtime as symbols only describe operations and not the underlying data they operate on. One workaround is to pass the type as a constructor argument of your network and hence build the appropriate compute graph for each situation.

### Compute Context

The same is true with the compute context and for the same reasons you cannot do:

```python
def hybrid_forward(self, F, x):
    if x.context == mx.cpu():
        return x
    return x*2
```

Without getting a `AttributeError: 'Symbol' object has no attribute 'context'`

Accessing the current compute context is not possible with symbols. Consider passing this information in the constructor if you require it to create the appropriate compute graph.

### Shape

Accessing shape information of tensors is very often used for example when trying to flatten a tensor and then reshape it back to its original shape.

```python
def hybrid_forward(self, F, x):
    return x*x.shape[0]
```

Trying to access the shape of a tensor in a hybridized block would get you this error: `AttributeError: 'Symbol' object has no attribute 'shape'`

Again, you cannot use the shape of the symbol at runtime as symbols only describe operations and not the underlying data they operate on. This will change in the future as MXNet will support [dynamic shape inference](https://cwiki.apache.org/confluence/display/MXNET/Dynamic+shape), and the shapes of symbols will be symbols themselves 

There are also a lot of operators that support special indices to help with most of the use-cases where you would want to access the shape information. For example, `F.reshape(x, (0,0,-1))` will keep the first two dimensions unchanged and collapse all further dimensions into the third dimension. See the documentation of the [`F.reshape`](http://beta.mxnet.io/api/ndarray/_autogen/mxnet.ndarray.reshape.html) for more details.

### Item assignment

Last but not least, you cannot directly assign values in tensor in a symbolic graph, the resulting tensors always needs to be the results of operations performed on the inputs of the computational graph. The following code:

```python
def hybrid_forward(self, F, x):
    x[0] = 2
    return x
```

Would get you this error `TypeError: 'Symbol' object does not support item assignment`

Direct item assignment is not possible in symbolic graph since it needs to be part of a computational graph. One way is to use add more inputs to your graph and use masking or the [`F.where`](http://beta.mxnet.io/api/ndarray/_autogen/mxnet.ndarray.where.html) operator.
