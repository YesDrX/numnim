import ../common
import strformat

type
  BlasVector*[T:SomeNumber] = ref object
    data*: ptr T #pointer to the first element
    size*: cint #length of vector
    stride*: cint #spacing between elements in vector, INCX INCY in blas.
    data_buffer*: seq[T] #so we can convert back to nim
    offset*: cint #offset of first elements in data_buffer
    itembytes*: cint #bytes taken by each element
    dtype*: string #data type
    has_nim_seq_data_buffer*: bool

  FourBytesType* = float32|int32|uint32|cfloat|cint|cuint
  EightBytesType* = float|float64|int64|uint|uint64|clong|culong|csize|csize_t|clonglong|cdouble|clongdouble|culonglong

proc init_BlasVector*[T](typeinfo: typedesc[T]): BlasVector[T]=
  result = new BlasVector[T]
  result.dtype = $T
  result.itembytes = T.sizeof.cint
  result.has_nim_seq_data_buffer = false

proc toBlasVector*[T](input: seq[T]): BlasVector[T] =
  result = init_BlasVector(T)
  result.data = unsafeAddr(input[0])
  result.size = input.len.cint
  result.stride = 1.cint
  result.offset = 0.cint
  result.has_nim_seq_data_buffer = true
  shallowCopy(result.data_buffer, input)

proc toBlasVector*[T](input: seq[T], stride: int, size: int, offset: int): BlasVector[T] =
  result = init_BlasVector(T)
  result.size = size.cint
  result.data = unsafeAddr(input[offset])
  result.stride = stride.cint
  result.offset = offset.cint
  result.has_nim_seq_data_buffer = true
  shallowCopy(result.data_buffer, input)  

proc toSeq*[T](input: BlasVector[T]): seq[T] =
  assert(input.has_nim_seq_data_buffer)
  result = newSeq[T](input.size.int)
  for idx in 0 .. (input.size - 1):
    result[idx] = input.data_buffer[idx * input.stride + input.offset]

proc forceCast*[T;U](input: BlasVector[T], output_type: typedesc[U]): BlasVector[U]=
  assert(T.sizeof == U.sizeof, fmt"byte size are mismatch when starting force casting.")
  result = init_BlasVector(U)
  result.data = cast[ptr U](input.data)
  result.size = input.size
  result.stride = input.stride
  result.offset = input.offset
  result.data_buffer = @[]
  result.has_nim_seq_data_buffer = false

proc `$`*[T](input: BlasVector[T]): string=
  assert(input.has_nim_seq_data_buffer, fmt"Your blas vector does not have nim sequence data buffer, this might be due to that you force casted this vector.")
  if input.size.int > 0:
    if input.size.int <= 10:
      result = printNdSeq(input.toSeq, @[input.size.int]) & fmt", shape = [{input.size.int}], BLAS Vector."
    else:
      result = "["
      for idx in 0..4:
        result &= `$`(input.data_buffer[input.stride * idx + input.offset]) & ", "
      result &= "..., "
      for idx in (input.size-5)..(input.size-2):
        result &= `$`(input.data_buffer[input.stride * idx + input.offset]) & ", "
      result &= fmt"], shape=[{input.size}]."
    if input.stride>1: result &= " Memory is not contiguous (Maybe casted from a ndarray view.)"
  else:
    result = fmt"[], Empty {input.dtype} Blas Vector."

when isMainModule:
  var
    a = xrange(100)
    b = a.toBlasVector
    c = b.forceCast(cdouble)
    d = c.forceCast(clong)
    e = d.forceCast(int)
  
  echo a[0].addr.repr
  echo b.data.repr
  echo c.data.repr
  echo d.data.repr
  echo e.data.repr

  echo b
  # echo c