import sequtils
import sugar
import strformat
import ../blas/blas_matrix
import ../blas/blas_vector
import nimblas/cblas
import ../common
import algorithm
# import special_ndarray

type
  NdArrayFlags* = ref object
    C_Continuous*: bool # = true # row major ndarray
    F_Continuous*: bool # = false
    isReadOnly*: bool # = false
    isCopiedFromView*: bool # = false

  NdArray*[T] {.shallow.} = ref object
    data_ptr* : ptr T
    data_buffer* : seq[T] # you may convert the pointer to c array above to a nim sequence using proc provided below, but it's better to carry the original sequence around, without copy of memory.
    start_idx_in_buffer* : int
    strides* : seq[int]
    shape* : seq[int]
    dtype* : string
    itembytes* : int
    flags* : NdArrayFlags

proc memory_is_continuous*[T](input: NdArray[T]): bool=
  if input.shape.len == 1 or input.shape.len > 2 or (input.shape.len == 2  and input.flags.C_Continuous):
    result = get_strides_from_shape(input.shape) == input.strides
  else:
    result = (input.strides[1] == input.shape[0] and input.strides[0] == 1)

proc nbytes*[T](input: NdArray[T]): int {.inline.} =
  result = input.itembytes * prodOfSeq(input.shape)

proc getIdx*[T](input: NdArray[T], idx: varargs[int]): int {.inline.} =
  assert(idx.len==input.strides.len, fmt"index {idx} can not access ndarray with shape {input.shape}.")
  result = 0
  for axis, axis_loc in idx:
    result += input.strides[axis] * axis_loc

proc at*[T](input: NdArray[T], idx: varargs[int]): T =
  result = input.data_buffer[input.getIdx(idx)]

proc set_at*[T](input: NdArray[T], value: T, idx: varargs[int]) =
  input.data_buffer[input.getIdx(idx)] = value

proc init_NdArrayFlags*(): NdArrayFlags=
  result = new NdArrayFlags
  result.C_Continuous = true
  result.F_Continuous = false
  result.isReadOnly = false
  result.isCopiedFromView = false

proc init_NdArray*[T](base_data_ptrtype: typedesc[T]): NdArray[T]=
  result = new NdArray[T]
  result.dtype = $T
  result.itembytes = T.sizeof
  result.flags = init_NdArrayFlags()

proc init_NdArray_from_existing*[T](input: NdArray[T]): NdArray[T]=
  result = init_NdArray(T)
  result.data_ptr = input.data_ptr
  shallowCopy(result.data_buffer, input.data_buffer)
  result.start_idx_in_buffer = input.start_idx_in_buffer
  result.shape = input.shape
  result.strides = input.strides
  result.flags = deepCopy(input.flags)
  result.dtype = input.dtype
  result.itembytes = input.itembytes

proc clone*[T](input: NdArray[T]): NdArray[T]=
  result = init_NdArray_from_existing(input)

proc copy*[T](input: NdArray[T]): NdArray[T]=
  var
    data_buffer = input.toSeq
  result = data_buffer.toNdArray(input.shape)

proc arange*[T:byte|int|int64|float|float32|cfloat|cdouble](start: T, stop: T, step: T = 1.T): NdArray[T]=
  var data_buffer : seq[T] = xrange(start, stop, step)
  result = data_buffer.toNdArray

proc arange*[T:byte|int|int64|float|float32|cfloat|cdouble](stop: T): NdArray[T]=
  result = arange(0.T, stop, 1.T)

proc get_location_from_coordinate(offset: int, strides: seq[int], coordinate: seq[int]): int=
  assert(strides.len == coordinate.len, fmt"invalid coordinate for strides.")
  result = offset
  for axis in 0..<strides.len:
    result += strides[axis] * coordinate[axis]

proc toSeq*[T](input: NdArray[T]): seq[T]=
  # Convert a ndarray data_ptr buffer to a nim sequence.
  if memory_is_continuous(input) and input.flags.C_Continuous:
    var
      first_coordinate = newSeq[int](input.shape.len)
      last_coordinate = newSeq[int](input.shape.len)
      first_location, last_location : int
    for axis in 0 ..< input.shape.len:
      last_coordinate[axis] = input.shape[axis] - 1
    first_location = get_location_from_coordinate(input.start_idx_in_buffer, input.strides, first_coordinate)
    last_location = get_location_from_coordinate(input.start_idx_in_buffer, input.strides, last_coordinate)
    if last_location - first_location + 1 == input.data_buffer.len:
      shallowCopy(result, input.data_buffer)
    else:
      result = input.data_buffer[first_location .. last_location]
  else:
    result = newSeq[T](prodOfSeq(input.shape))
    var axis_idx : seq[seq[int]] = @[]
    for i in 0 .. input.shape.len-1:
      axis_idx.add(arange(input.shape[i]).data_buffer)
    var all_idx = product(axis_idx)
    all_idx.sort(cmp_seq_lexical)
    var idx_in_data_buffer: int
    if input.shape.len > 1:
      for i, item_idx in all_idx:
        idx_in_data_buffer = get_idx(input, item_idx) + input.start_idx_in_buffer
        result[i] = input.data_buffer[idx_in_data_buffer]
    else:
      for i, item_idx in all_idx[0]:
        idx_in_data_buffer = get_idx(input, item_idx) + input.start_idx_in_buffer
        result[i] = input.data_buffer[idx_in_data_buffer]

proc toBlasVector*[T](input: NdArray[T]): BlasVector[T]=
  assert(input.shape.len == 1, fmt"input ndarray is not a vector, so it cannot be converted to BlasVector.")
  result = init_BlasVector(T.type)
  result.data = input.data_ptr
  result.size = input.shape[0].cint
  result.stride = input.strides[0].cint
  result.offset = input.start_idx_in_buffer.cint
  result.has_nim_seq_data_buffer = true
  shallowCopy(result.data_buffer, input.data_buffer)

proc toBlasMatrix*[T](input: NdArray[T]): BlasMatrix[T]=
  assert(input.shape.len == 2, fmt"input ndarray is not a matrix, so it cannot be converted to BlasMatrix.")
  assert(input.strides[0] == 1 or input.strides[1] == 1, fmt"can not determine if the input matrix is row major or column major.")
  result = init_BlasMatrix(T.type)
  result.data = input.data_ptr
  result.m = input.shape[0].cint
  result.n = input.shape[1].cint
  result.has_nim_seq_data_buffer = true

  if input.strides == @[1,1]:
    if input.shape[0] == 1:
      result.lda = 1.cint
      result.order = CBLAS_ORDER.CblasRowMajor
    else:
      result.lda = 1.cint
      result.order = CBLAS_ORDER.CblasColMajor
  elif input.strides[^1] == 1:
    assert(input.flags.C_Continuous)
    result.lda = input.strides[0].cint
    result.order = CBLAS_ORDER.CblasRowMajor
  elif input.strides[0] == 1:
    assert(input.flags.F_Continuous)
    result.lda = input.strides[1].cint
    result.order = CBLAS_ORDER.CblasColMajor

  shallowCopy(result.data_buffer, input.data_buffer)
  result.offset = input.start_idx_in_buffer.cint

proc toNdArray*[T](input: seq[seq[T]]): auto=
  let implied_shape: seq[int] = input.getShapeOfSeq
  let flatten_data_ptr = input.flattenSeq
  result = toNdArray(flatten_data_ptr, implied_shape)

proc toNdArray*[T](input: seq[T]): NdArray[T]=
  result = init_NdArray(T)
  result.data_buffer = input
  if input.len>0: result.data_ptr = addr(result.data_buffer[0])
  result.strides = @[1]
  result.shape = @[input.len]

proc toNdArray*[T](input: seq[T], shape: seq[int]): NdArray[T]=
  if input.len>0:
    assert(prodOfSeq(shape)==input.len, fmt"shape is not compatible with data_ptr length prod({shape})(={prodOfSeq(shape)})!={input.len}.")
  result = toNdArray(input)
  result.strides =  get_strides_from_shape(shape)
  result.shape = shape

proc toNdArray*[T](input: var seq[T], shape: openArray[int]): NdArray[T]=
  result = input.toNdArray(shape.toSeq)

proc toNdArray*[T:cfloat|cdouble](input: BlasVector[T]): NdArray[T]=
  result = init_NdArray(T)
  result.data_ptr = input.data
  shallowCopy(result.data_buffer, input.data_buffer)
  result.shape = @[input.size.int]
  result.strides = @[input.stride.int]
  result.start_idx_in_buffer = input.offset.int

proc toNdArray*[T:cfloat|cdouble](input: BlasVector[T], shape: seq[int]): NdArray[T]=
  assert(prodOfSeq(shape) == (input.data_buffer.len - input.offset.int) div input.stride.int, fmt"shape={shape} is not compatible with input Blas vector of length {input.data_buffer.len}!")
  result =toNdArray(input)
  if input.stride.int == 1:
    result.shape = shape
    result.strides = get_strides_from_shape(shape)
  else:
    result = result.toSeq.toNdArray
    result.shape = shape
    result.strides = get_strides_from_shape(shape)

proc toNdArray*[T:cfloat|cdouble](input: BlasVector[T], shape: openArray[int]): NdArray[T]=
  result = toNdArray(input, shape.toSeq)

proc toNdArray*[T:cfloat|cdouble](input: BlasMatrix[T]): NdArray[T]=
  result = init_NdArray(T)
  result.data_ptr = input.data
  shallowCopy(result.data_buffer, input.data_buffer)
  if input.order == CBLAS_ORDER.CblasRowMajor:
    result.strides = @[input.lda.int,1]
  else:
    result.strides = @[1, input.lda.int]
    result.flags.F_Continuous = true
    result.flags.C_Continuous = false
  result.shape = @[input.m.int, input.n.int]
  result.start_idx_in_buffer = input.offset.int

proc `$`*(flags: NdArrayFlags): string=
  result = "C_Continuous: " & $flags.C_Continuous & "\n" &
           "F_Continuous: " & $flags.F_Continuous & "\n" &
           "isReadOnly: " & $flags.isReadOnly & "\n" &
           "isCopiedFromView: " & $flags.isCopiedFromView

proc `$`*[T](input: NdArray[T]): string=
  if memory_is_continuous(input) and input.flags.C_Continuous:
    var
      start_idx = input.start_idx_in_buffer
      end_idx = start_idx + prodOfSeq(input.shape) - 1
    result = printNdSeq(input.data_buffer[start_idx .. end_idx], input.shape) & ", shape = " & printNdSeq(input.shape,@[input.shape.len])
  elif memory_is_continuous(input) and input.flags.F_Continuous:
    var
      #TODO: use ravel function
      data_buffer = input.toSeq
    result = printNdSeq(data_buffer, input.shape) & ", shape = " & printNdSeq(input.shape,@[input.shape.len]) & ", memory is F_Continuous."
  else:
    #TODO: the following creation of temp data_buffer is not neccesary!
    var
      #TODO: use ravel function
      data_buffer = input.toSeq
    result = printNdSeq(data_buffer, input.shape) & ", shape = " & printNdSeq(input.shape,@[input.shape.len]) & ", memory is not contiguous (maybe a view)."

when isMainModule:
  # var a = arange(12)
  # echo a
  # var b = a.toSeq.astype(cfloat)
  # echo b
  # var c = b.toBlasVector
  # echo c
  # c.stride = 2.cint
  # c.size = 6.cint
  # echo c
  # var d = c.toNdArray
  # echo d
  # var
  #   a = arange(120).toSeq
  #   b = a.toNdArray(@[2,3,4,5])

  # echo b
  # echo b.strides

  # b.start_idx_in_buffer = 20
  # b.shape = @[1,2,4,5]
  # echo b
  # 

  var
    a = init_NdArrayFlags()
    b = a
    c = deepCopy(a)
    B = @[@[-1.14318735,  0.82482718,  0.59112794,  0.10233465, -1.499152  ],
          @[ 1.06265695,  0.94507383,  1.24635915,  1.66620749,  0.86445065],
          @[-0.60062639,  0.46063082,  1.21422265,  0.74385278, -0.40165673]].flattenSeq.toNdArray(@[3,5])
  a.isReadOnly = true
  var d = deepCopy(a)
  echo a.isReadOnly.addr.repr
  echo b.isReadOnly.addr.repr
  echo c.isReadOnly.addr.repr
  echo d.isReadOnly.addr.repr

  echo a.isReadOnly
  echo b.isReadOnly
  echo c.isReadOnly
  echo d.isReadOnly

  echo B