import ndarray
import sequtils
import math
import ../common

proc empty*[T](shape: seq[int], typename: typedesc[T]): NdArray[T]=
  var data_buffer = newSeq[T](prodOfSeq(shape))
  result = data_buffer.toNdArray(shape)

proc empty*[T](shape: openArray[int], typename: typedesc[T]): NdArray[T]=
  result = empty(shape.toSeq, typename)

proc zeros*(length: int): NdArray[float]=
  result = empty(@[length], float)

proc zeros*(shape: seq[int]): NdArray[float]=
  result = empty(shape, float)

proc zeros*(shape: openArray[int]): NdArray[float]=
  result = empty(shape.toSeq, float)

proc nans*(length: int): NdArray[float]=
  var data_buffer = newSeqWith[float](length, NaN)
  result = data_buffer.toNdArray

proc nans*(shape: seq[int]): NdArray[float]=
  var data_buffer = newSeqWith[float](prodOfSeq(shape), NaN)
  result = data_buffer.toNdArray(shape)

proc nans*(shape: openArray[int]): NdArray[float]=
  result = nans(shape.toSeq)

proc ones*(length: int): NdArray[float]=
  var data_buffer = newSeqWith[float](length, 1.0)
  result = data_buffer.toNdArray

proc ones*(shape: seq[int]): NdArray[float]=
  var data_buffer = newSeqWith[float](prodOfSeq(shape), 1.0)
  result = data_buffer.toNdArray(shape)

proc ones*(shape: openArray[int]): NdArray[float]=
  result = ones(shape.toSeq)

proc eye*(n: int): NdArray[float]=
  result = empty(@[n,n],float)
  for idx in 0..n-1:
    result.data_buffer[idx*n+idx] = 1.0

proc diag*[T](diagnal_values: seq[T]): NdArray[T]=
  var n = diagnal_values.len
  result = empty(@[n,n],T)
  for idx in 0..n-1:
    result.data_buffer[idx*n+idx] = diagnal_values[idx]

proc diag*[T](diagnal_values: openArray[T]): NdArray[T]=
  result = diag(diagnal_values.toSeq)

proc diag*[T](mat: NdArray[T]): NdArray[T]=
  assert(mat.strides.len <= 2, "input is not a matrix or a vector.")
  if mat.strides.len == 2:
    var
      n = mat.shape[0]
    if mat.shape[1] < n:
      n = mat.shape[1]
    var data_buffer = newSeq[T](n)
    for idx in 0..n-1:
      data_buffer[idx] = mat.at(idx,idx)
    result = data_buffer.toNdArray
  else:
    result = mat.data_buffer.diag

proc linspace*[T](start, stop:T, num: int): NdArray[T]=
  var
    step = (stop - start) / (num.T-1)
    data_buffer = xrange(start, stop, step, inclusive=true)
  result = data_buffer.toNdArray

proc logspace*[T:float|float32|cfloat|cdouble|clongdouble](start, stop:T, num: int, base: float = 10.0): NdArray[T]=
  result = linspace(start,stop,num)
  for i in 0..<result.shape[0]:
    result.data_buffer[i] = exp(result.data_buffer[i] * ln(base))

when isMainModule:
  echo ones([10,10])
  echo diag([1,2,3])
  echo diag([1,2,3]).diag
  echo arange(10)
  echo linspace(0.0, 7.0, 10)
  echo linspace(0.0, 9.0, 10)
  echo logspace(2.0, 3.0, 4)