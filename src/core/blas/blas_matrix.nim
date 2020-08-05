import ../common
import nimblas/cblas
import strformat

type
  BlasMatrix*[T:SomeNumber] = ref object
    data* : ptr T
    m* : cint # number of rows
    n* : cint # number of columns
    lda* : cint # stride between consecutive elements of the same row (for col major matrix); or stride between consecutive elements of the same column (for row major matrix)
    order* : CBLAS_ORDER # C_Continuous (lda=m) or F_Continuous (lda=n)
    data_buffer* : seq[T]
    offset*: cint
    has_nim_seq_data_buffer*: bool

proc init_BlasMatrix*[T](typeinfo: typedesc[T]): BlasMatrix[T]=
  result = new BlasMatrix[T]
  result.has_nim_seq_data_buffer = false

proc toBlasMatrix*[T: cfloat|cdouble](input : seq[T], m : cint|int|int64, n : cint|int|int64) : BlasMatrix[T]=
  result = new BlasMatrix[T]
  result.data = unsafeAddr(input[0])
  result.m = m.cint
  result.n = n.cint
  result.lda = n.cint
  result.order = CBLAS_ORDER.CblasRowMajor
  result.offset = 0.cint
  result.has_nim_seq_data_buffer = true
  shallowCopy(result.data_buffer, input)

proc toSeq*(input: BlasMatrix[cfloat]) : seq[float32]=
  if input.order == CBLAS_ORDER.CblasRowMajor:
    if (input.m * input.n).int < input.data_buffer.len:
      result = newSeq[float32]((input.m * input.n).int)
      var idx_in_data_buffer: int
      for rowIdx in 0..(input.m.int-1):
        for colIdx in 0..(input.n.int-1):
          idx_in_data_buffer = rowIdx * input.lda.int + colIdx
          result[rowIdx * input.n.int + colIdx] = input.data_buffer[idx_in_data_buffer]
    else:
      return input.data_buffer.astype(float32)
  else:
    result = newSeq[float32]((input.m * input.n).int)
    var idx_in_data_buffer: int
    for rowIdx in 0..(input.m.int-1):
      for colIdx in 0..(input.n.int-1):
        idx_in_data_buffer = rowIdx + input.lda.int * colIdx
        result[rowIdx * input.n.int + colIdx] = input.data_buffer[idx_in_data_buffer]

proc toSeq*(input: BlasMatrix[cdouble]) : seq[float]=
  if input.order == CBLAS_ORDER.CblasRowMajor:
    if (input.m * input.n).int < input.data_buffer.len:
      result = newSeq[float]((input.m * input.n).int)
      var idx_in_data_buffer: int
      for rowIdx in 0..(input.m.int-1):
        for colIdx in 0..(input.n.int-1):
          idx_in_data_buffer = rowIdx * input.lda.int + colIdx
          result[rowIdx * input.n.int + colIdx] = input.data_buffer[idx_in_data_buffer]
    else:
      return input.data_buffer.astype(float)
  else:
    result = newSeq[float]((input.m * input.n).int)
    var idx_in_data_buffer: int
    for rowIdx in 0..(input.m.int-1):
      for colIdx in 0..(input.n.int-1):
        idx_in_data_buffer = rowIdx + input.lda.int * colIdx
        result[rowIdx * input.n.int + colIdx] = input.data_buffer[idx_in_data_buffer]

proc forceCast*[T;U](input: BlasMatrix[T], output_type: typedesc[U]): BlasMatrix[U]=
  assert(T.sizeof == U.sizeof, fmt"byte size are mismatch when starting force casting.")
  result = init_BlasMatrix(U)
  result.data = cast[ptr U](input.data)
  result.m = input.m
  result.n = input.n
  result.lda = input.lda
  result.order = input.order
  result.offset = input.offset
  result.data_buffer = @[]
  result.has_nim_seq_data_buffer = false

proc `$`*[T](input: BlasMatrix[T]): string=
  assert(input.has_nim_seq_data_buffer, fmt"Your blas matrix does not have nim sequence data buffer, this might be due to that you force casted this matrix.")
  result = `$`(input.toNdArray) & " BLAS Matrix."

when isMainModule:
  var
    a = @[1,2,3,4,5,6,8,9,10,11,12].astype(cfloat)
    b = a.toBlasMatrix(3,4)
  echo b.m
  echo b.n
  echo b.toSeq