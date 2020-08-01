import ndarray
import ndarray_indexers
import sequtils
import strformat
import ../blas/blas_vector
import ../blas/blas_matrix
from ../blas/blas_wrapper as blas import nil
import nimblas/cblas
from nimblas/cblas as cblas import nil
from nimlapack as lapack import nil
import ../common
import algorithm


proc ndarray_memory_block_to_blas_vector[T](input:NdArray[T], offset: int, stride:int, length: int): BlasVector[T]=
  result = init_BlasVector(T)
  result.data = addr(input.data_buffer[offset])
  shallowCopy(result.data_buffer, input.data_buffer)
  result.stride = stride.cint
  result.offset = offset.cint
  result.size = length.cint
  result.has_nim_seq_data_buffer = true

# AXPY: y = alpha * x + y
# proc axpy*(x : BlasVector[cfloat], y : BlasVector[cfloat], alpha: float32|float|cfloat|cdouble) =
proc AXPY_Helper[T](A, B: NdArray[T], op: string): NdArray[T]=
  assert(A.shape == B.shape, fmt"We do not support broadcasting yet.")
  assert(T.sizeof == 4 or T.sizeof == 8, fmt"Wrong data type.")

  var
    C = A.copy
    (coordinates, coordinates_copy, memory_blocks_axis, block_length) = get_memory_blocks(C.shape, get_indexer_list(C.shape))
    strideC, strideB: int
    offsetC, offsetB: int
    y_vector, x_vector: BlasVector[T]
    y_vector_cfloat, x_vector_cfloat : BlasVector[cfloat]
    y_vector_cdouble, x_vector_cdouble : BlasVector[cdouble]
    alpha : float= 0.0

  if op == "+":
    alpha = 1.0
  elif op == "-":
    alpha = -1.0

  strideC = C.strides[memory_blocks_axis]
  strideB = B.strides[memory_blocks_axis]

  if T.sizeof == 4:
    for i in 0..<coordinates.len:
      offsetC = get_location_from_coordinate(C.start_idx_in_buffer, C.strides, coordinates[i])
      offsetB = get_location_from_coordinate(B.start_idx_in_buffer, B.strides, coordinates[i])

      y_vector = ndarray_memory_block_to_blas_vector(C, offsetC, strideC, block_length)
      x_vector = ndarray_memory_block_to_blas_vector(B, offsetB, strideB, block_length)

      y_vector_cfloat = y_vector.force_cast(cfloat)
      x_vector_cfloat = x_vector.force_cast(cfloat)

      blas.axpy(x_vector_cfloat, y_vector_cfloat, alpha)
  else:
    for i in 0..<coordinates.len:
      offsetC = get_location_from_coordinate(C.start_idx_in_buffer, C.strides, coordinates[i])
      offsetB = get_location_from_coordinate(B.start_idx_in_buffer, B.strides, coordinates[i])

      y_vector = ndarray_memory_block_to_blas_vector(C, offsetC, strideC, block_length)
      x_vector = ndarray_memory_block_to_blas_vector(B, offsetB, strideB, block_length)

      y_vector_cdouble = y_vector.force_cast(cdouble)
      x_vector_cdouble = x_vector.force_cast(cdouble)

      blas.axpy(x_vector_cdouble, y_vector_cdouble, alpha)

  result = C

proc AXPY_Scalar_Helper[T;U](A: NdArray[T], b:U, op: string): NdArray[T]=
  assert(T.sizeof == 4 or T.sizeof == 8, fmt"Wrong data type.")

  var
    B = b.T
    C = A.copy
    (coordinates, coordinates_copy, memory_blocks_axis, block_length) = get_memory_blocks(C.shape, get_indexer_list(C.shape))
    strideC, strideB: int
    offsetC, offsetB: int
    y_vector, x_vector: BlasVector[T]
    y_vector_cfloat, x_vector_cfloat : BlasVector[cfloat]
    y_vector_cdouble, x_vector_cdouble : BlasVector[cdouble]
    alpha : float= 0.0
    data_buffer_B: seq[T]

  if op == "+":
    alpha = 1.0
  elif op == "-":
    alpha = -1.0

  strideC = C.strides[memory_blocks_axis]

  data_buffer_B = newSeq[T](block_length)
  for i in 0 ..< data_buffer_B.len: data_buffer_B[i] = B
  x_vector = data_buffer_B.toBlasVector
  strideB = 1
  
  if T.sizeof == 4:
    x_vector_cfloat = x_vector.force_cast(cfloat)
    for i in 0..<coordinates.len:
      offsetC = get_location_from_coordinate(C.start_idx_in_buffer, C.strides, coordinates[i])
      y_vector = ndarray_memory_block_to_blas_vector(C, offsetC, strideC, block_length)
      y_vector_cfloat = y_vector.force_cast(cfloat)
      blas.axpy(x_vector_cfloat, y_vector_cfloat, alpha)
  else:
    x_vector_cdouble = x_vector.force_cast(cdouble)
    for i in 0..<coordinates.len:
      offsetC = get_location_from_coordinate(C.start_idx_in_buffer, C.strides, coordinates[i])
      y_vector = ndarray_memory_block_to_blas_vector(C, offsetC, strideC, block_length)
      y_vector_cdouble = y_vector.force_cast(cdouble)
      blas.axpy(x_vector_cdouble, y_vector_cdouble, alpha)

  result = C

proc `+`*[T](A, B: NdArray[T]): NdArray[T]=
  result = AXPY_Helper(A,B,"+")

proc `+`*[T;U](A: NdArray[T], B:U): NdArray[T]=
  result = AXPY_Scalar_Helper(A,B,"+")

proc `+`*[T;U](B:U, A: NdArray[T]): NdArray[T]=
  result = AXPY_Scalar_Helper(A,B,"+")

proc `-`*[T](A, B: NdArray[T]): NdArray[T]=
  result = AXPY_Helper(A,B,"-")

proc `-`*[T;U](A: NdArray[T], B:U): NdArray[T]=
  result = AXPY_Scalar_Helper(A,B,"-")

proc `-`*[T;U](B:U, A: NdArray[T]): NdArray[T]=
  result = AXPY_Scalar_Helper(A*(-1),B,"+")

proc `-`*[T](A: NdArray[T]): NdArray[T]=
  result = AXPY_Scalar_Helper(A*(-1),0,"+")

proc element_wise_mult_blas_vector[T](x, y: BlasVector[T]):BlasVector[T] =
  # z = x element_wise_multiply_with y
  # using SBMV(Uplo, alpha, K, beta, A, x, y) : y = alpha * A * x + beta*y
  # let K=0, alpha=1, beta=0, A=diag(y) (band format), x = x, then result will be x.*y output to y.
  let
    Uplo = CBLAS_UPLO.CblasLower
    K = 0.cint
    alpha = 1.0
    beta = 0.0
    A = new BlasMatrix[T]
  result = newSeq[T](x.size).toBlasVector

  A.data = y.data
  A.m = y.size
  A.n = y.size
  A.lda = y.stride
  A.order = CBLAS_ORDER.CblasRowMajor
  blas.sbmv(Uplo, alpha, K, beta, A, x, result)

proc element_wise_div_blas_vector[T](x, y: BlasVector[T]) =
  # x = x / y
  # using TBSV(Uplo, TransA, Diag, A, K, x) : x = A^-1 * x
  # let Uplo = CBLAS_UPLO.CblasLower, TransA
  let
    Uplo = CBLAS_UPLO.CblasLower
    TransA = CBLAS_TRANSPOSE.CblasNoTrans
    Diag = CBLAS_DIAG.CblasNonUnit
    K = 0.cint
    A = new BlasMatrix[T]

  A.data = y.data
  A.m = y.size
  A.n = y.size
  A.lda = y.stride
  A.order = CBLAS_ORDER.CblasRowMajor

  blas.tbsv(Uplo, TransA, Diag, A, K, x)

proc `*`*[T](A, B: NdArray[T]): NdArray[T]=
  assert(A.shape == B.shape, fmt"We do not support broadcasting yet.")
  assert(T.sizeof == 4 or T.sizeof == 8, fmt"Wrong data type.")

  var
    C = A.copy
    (coordinates, coordinates_copy, memory_blocks_axis, block_length) = get_memory_blocks(C.shape, get_indexer_list(C.shape))
    strideC, strideB: int
    offsetC, offsetB: int
    y_vector, x_vector: BlasVector[T]
    y_vector_cfloat, x_vector_cfloat, z_vector_cfloat : BlasVector[cfloat]
    y_vector_cdouble, x_vector_cdouble, z_vector_cdouble : BlasVector[cdouble]

  strideC = C.strides[memory_blocks_axis]
  strideB = B.strides[memory_blocks_axis]

  if T.sizeof == 4:
    for i in 0..<coordinates.len:
      offsetC = get_location_from_coordinate(C.start_idx_in_buffer, C.strides, coordinates[i])
      offsetB = get_location_from_coordinate(B.start_idx_in_buffer, B.strides, coordinates[i])

      y_vector = ndarray_memory_block_to_blas_vector(C, offsetC, strideC, block_length)
      x_vector = ndarray_memory_block_to_blas_vector(B, offsetB, strideB, block_length)

      y_vector_cfloat = y_vector.force_cast(cfloat)
      x_vector_cfloat = x_vector.force_cast(cfloat)

      z_vector_cfloat = element_wise_mult_blas_vector(x_vector_cfloat, y_vector_cfloat)
      blas.swap(y_vector_cfloat, z_vector_cfloat)

  else:
    for i in 0..<coordinates.len:
      offsetC = get_location_from_coordinate(C.start_idx_in_buffer, C.strides, coordinates[i])
      offsetB = get_location_from_coordinate(B.start_idx_in_buffer, B.strides, coordinates[i])

      y_vector = ndarray_memory_block_to_blas_vector(C, offsetC, strideC, block_length)
      x_vector = ndarray_memory_block_to_blas_vector(B, offsetB, strideB, block_length)

      y_vector_cdouble = y_vector.force_cast(cdouble)
      x_vector_cdouble = x_vector.force_cast(cdouble)

      z_vector_cdouble = element_wise_mult_blas_vector(x_vector_cdouble, y_vector_cdouble)
      blas.swap(y_vector_cdouble, z_vector_cdouble)

  result = C

proc Multiply_With_Scalar[T](A:NdArray[T], B:T): NdArray[T]=
  assert(T.sizeof == 4 or T.sizeof == 8, fmt"Wrong data type.")

  var
    C = A.copy
    (coordinates, coordinates_copy, memory_blocks_axis, block_length) = get_memory_blocks(C.shape, get_indexer_list(C.shape))
    strideC, strideB: int
    offsetC, offsetB: int
    y_vector, x_vector: BlasVector[T]
    y_vector_cfloat, x_vector_cfloat, z_vector_cfloat : BlasVector[cfloat]
    y_vector_cdouble, x_vector_cdouble, z_vector_cdouble : BlasVector[cdouble]
    data_buffer_B: seq[T]

  strideC = C.strides[memory_blocks_axis]
  data_buffer_B = newSeq[T](block_length)
  for i in 0 ..< data_buffer_B.len: data_buffer_B[i] = B
  x_vector = data_buffer_B.toBlasVector
  strideB = 1

  if T.sizeof == 4:
    x_vector_cfloat = x_vector.force_cast(cfloat)
    for i in 0..<coordinates.len:
      offsetC = get_location_from_coordinate(C.start_idx_in_buffer, C.strides, coordinates[i])
      y_vector = ndarray_memory_block_to_blas_vector(C, offsetC, strideC, block_length)
      y_vector_cfloat = y_vector.force_cast(cfloat)
      z_vector_cfloat = element_wise_mult_blas_vector(x_vector_cfloat, y_vector_cfloat)
      blas.swap(y_vector_cfloat, z_vector_cfloat)
  else:
    x_vector_cdouble = x_vector.force_cast(cdouble)
    for i in 0..<coordinates.len:
      offsetC = get_location_from_coordinate(C.start_idx_in_buffer, C.strides, coordinates[i])
      y_vector = ndarray_memory_block_to_blas_vector(C, offsetC, strideC, block_length)
      y_vector_cdouble = y_vector.force_cast(cdouble)
      z_vector_cdouble = element_wise_mult_blas_vector(x_vector_cdouble, y_vector_cdouble)
      blas.swap(y_vector_cdouble, z_vector_cdouble)

  result = C

proc `*`*[T;U](A:NdArray[T], b:U): NdArray[T]=
  result = Multiply_With_Scalar(A,b.T)

proc `*`*[T;U](b:U,A:NdArray[T]): NdArray[T]=
  result = Multiply_With_Scalar(A,b.T)

proc `/`*[T;U](A:NdArray[T], b:U): NdArray[T]=
  result = Multiply_With_Scalar(A,1/(b.T))

proc `/`*[T](A, B: NdArray[T]): NdArray[T]=
  assert(A.shape == B.shape, fmt"We do not support broadcasting yet.")
  assert(T.sizeof == 4 or T.sizeof == 8, fmt"Wrong data type.")

  var
    C = A.copy
    (coordinates, coordinates_copy, memory_blocks_axis, block_length) = get_memory_blocks(C.shape, get_indexer_list(C.shape))
    strideC, strideB: int
    offsetC, offsetB: int
    y_vector, x_vector: BlasVector[T]
    y_vector_cfloat, x_vector_cfloat, z_vector_cfloat : BlasVector[cfloat]
    y_vector_cdouble, x_vector_cdouble, z_vector_cdouble : BlasVector[cdouble]

  strideC = C.strides[memory_blocks_axis]
  strideB = B.strides[memory_blocks_axis]

  if T.sizeof == 4:
    for i in 0..<coordinates.len:
      offsetC = get_location_from_coordinate(C.start_idx_in_buffer, C.strides, coordinates[i])
      offsetB = get_location_from_coordinate(B.start_idx_in_buffer, B.strides, coordinates[i])

      y_vector = ndarray_memory_block_to_blas_vector(C, offsetC, strideC, block_length)
      x_vector = ndarray_memory_block_to_blas_vector(B, offsetB, strideB, block_length)

      y_vector_cfloat = y_vector.force_cast(cfloat)
      x_vector_cfloat = x_vector.force_cast(cfloat)

      element_wise_div_blas_vector(y_vector_cfloat, x_vector_cfloat)

  else:
    for i in 0..<coordinates.len:
      offsetC = get_location_from_coordinate(C.start_idx_in_buffer, C.strides, coordinates[i])
      offsetB = get_location_from_coordinate(B.start_idx_in_buffer, B.strides, coordinates[i])

      y_vector = ndarray_memory_block_to_blas_vector(C, offsetC, strideC, block_length)
      x_vector = ndarray_memory_block_to_blas_vector(B, offsetB, strideB, block_length)

      y_vector_cdouble = y_vector.force_cast(cdouble)
      x_vector_cdouble = x_vector.force_cast(cdouble)

      element_wise_div_blas_vector(y_vector_cdouble, x_vector_cdouble)

  result = C

proc `/`*[T;U](b:U, A: NdArray[T]): NdArray[T]=
  var
    data_buffer = newSeq[T](prodOfSeq(A.shape))
    B = b.T
  for i in  0..<data_buffer.len: data_buffer[i] = B
  var C = data_buffer.toNdArray(A.shape)
  result = `/`(C,A)

proc transpose*[T](A:NdArray[T]): NdArray[T]=
  result = A.clone
  result.shape.reverse
  result.strides.reverse
  if result.flags.C_Continuous:
    result.flags.C_Continuous = false
    result.flags.F_Continuous = true
  else:
    result.flags.C_Continuous = true
    result.flags.F_Continuous = false
  result.flags.isReadOnly = true 
  result.flags.isCopiedFromView = true

proc dot*[T](A,B:NdArray[T]): NdArray[T]=
  assert(A.shape.len == 2 and B.shape.len == 2, fmt"Input ndarrys are not 2-dimensional, when calculating dot product.")
  assert(A.shape[1] == B.shape[0], fmt"Input matrics dimentions are not compatible for dot product. Left Shape = {A.shape}, Right Shape = {B.shape}")  
  assert(T.sizeof == 4 or T.sizeof == 8, fmt"Wrong data type.")
  var
    matA : BlasMatrix[T]
    matB : BlasMatrix[T]
    C =  newSeq[T](A.shape[0] * B.shape[1]).toNdArray(@[B.shape[1], A.shape[0]]).transpose
    matC = C.toBlasMatrix
    TransA, TransB: CBLAS_TRANSPOSE
    blasA_s, blasB_s, blasC_s : BlasMatrix[cfloat]
    blasA_d, blasB_d, blasC_d : BlasMatrix[cdouble]

  if A.flags.F_Continuous:
    matA = A.toBlasMatrix
    TransA = CBLAS_TRANSPOSE.CblasNoTrans
  else:
    matA = A.transpose.toBlasMatrix
    TransA = CBLAS_TRANSPOSE.CblasTrans
  matA.order = CBLAS_ORDER.CblasColMajor
  
  if B.flags.F_Continuous:
    matB = B.toBlasMatrix
    TransB = CBLAS_TRANSPOSE.CblasNoTrans
  else:
    matB = B.transpose.toBlasMatrix
    TransB = CBLAS_TRANSPOSE.CblasTrans
  matB.order = CBLAS_ORDER.CblasColMajor

  if T.sizeof == 4:
    blasA_s = matA.forceCast(cfloat)
    blasB_s = matB.forceCast(cfloat)
    blasC_s = matC.forceCast(cfloat)
    cblas.sgemm(blasA_s.order, TransA, TransB, blasC_s.m, blasC_s.n, blasA_s.m, 1.cfloat, blasA_s.data, blasA_s.lda, blasB_s.data, blasB_s.lda, 1.cfloat, blasC_s.data, blasC_s.lda)
  else:
    blasA_d = matA.forceCast(cdouble)
    blasB_d = matB.forceCast(cdouble)
    blasC_d = matC.forceCast(cdouble)
    cblas.dgemm(blasA_d.order, TransA, TransB, blasC_d.m, blasC_d.n, blasA_d.m, 1.cdouble, blasA_d.data, blasA_d.lda, blasB_d.data, blasB_d.lda, 1.cdouble, blasC_d.data, blasC_d.lda)

  result = C

proc reshape*[T](A:NdArray[T], shape: seq[int]): NdArray[T]=
  #for now, when reshape a ndarray, we make a clean copy
  var
    data_buffer = A.toSeq
  result = data_buffer.toNdArray(shape)

proc ravel*[T](A:NdArray[T]): NdArray[T]=
  var
    data_buffer = A.toSeq
  result = data_buffer.toNdArray

proc flatten*[T](A:NdArray[T]): NdArray[T]=
  result = A.ravel


when isMainModule:
  # var
  # #   B = @[1,2,3,4].astype(float).toNdArray(@[2,2])
  # #   (q,r) = B.qr
  #   A = @[1,2].astype(float).toNdArray(@[2,1])
  #   B = @[3,4].astype(float).toNdArray(@[1,2])
  # echo A.dot(B)
  var
    idx = get_indexer_list(@[1,10])
    blocks = get_memory_blocks(@[1,10], idx)
  echo blocks
