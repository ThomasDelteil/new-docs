# A Hybrid of Imperative and Symbolic Programming

## Pre-requisite

Understanding the difference between [Imperative and Symbolic Programming with MXNet](https://d2l.ai/chapter_computational-performance/hybridize.html) and how to create hybridized graphs. 

## Key differences and limitations of hybridized graph

### Indexing

```python
def hybrid_forward(self, F, x):
    return x[0,0]
```

`TypeError: Symbol only support integer index to fetch i-th output`

You cannot use multi-part indexing with Symbols. 
However you have a family of operators that can help you with array manipulations like: `nd.split`, `nd.slice`, `nd.take`, `nd.pick`, `nd.where`, `nd.reshape`, `nd.reshape_like`.

### Types

```python
def hybrid_forward(self, F, x):
    if x.dtype =='float16':
        return x
    return x*2
```

`AttributeError: 'Symbol' object has no attribute 'dtype'`

This is not possible in a hybridized graph. You cannot use the type of the symbol at runtime as symbols only describe operations but not the underlying data they operate on. One workaround is to pass the type as a constructor argument of your network and hence build the appropriate compute graph for each situation.

### Compute Context

```python
def hybrid_forward(self, F, x):
    if x.context == mx.cpu():
        return x
    return x*2
```
`AttributeError: 'Symbol' object has no attribute 'context'`

Similarly and for the same reasons, accessing the current compute context is not possible with symbols. Consider passing this information in the constructor if you require it to create the appropriate compute graph.

### Shape

```python
def hybrid_forward(self, F, x):
    return x*x.shape[0]
```

`AttributeError: 'Symbol' object has no attribute 'shape'`

Unfortunately this is not possible in a hybridized graph. You cannot use the shape of the symbol at runtime as symbols only describe operations but not the underlying data they operate on. This will change in the future as MXNet will support dynamic shape inference. https://cwiki.apache.org/confluence/display/MXNET/Dynamic+shape

There is also a lot of operator that support `magic` indices to help with most of the use-cases where you would want to use dynamic shapes. See the documentation of the `nd.reshape` operator for example.

### Item assignment


```python
def hybrid_forward(self, F, x):
    x[0] = 2
    return x
```

`TypeError: 'Symbol' object does not support item assignment`

Direct item assignment is not possible in symbolic graph since it needs to be part of a computational graph. One way is to use masking or the `nd.where` operator.
