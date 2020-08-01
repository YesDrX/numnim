import ../ndarray/ndarray
import ../ndarray/ndarray_indexers
import ../ndarray/ndarray_operators
import ../blas/blas_vector
import ../blas/blas_matrix
import ../ndarray/special_ndarray
import ../ndarray/accessers
import nimlapack
import ../ndarray/random_ndarray
import nimblas/cblas
import ../common
import sequtils
import strformat

proc det*[T: SomeFloat](A: NdArray[T]): float=
  # Use xGERF to calculate inverse of a matrix.
  assert(A.shape.len == 2 and A.shape[0] == A.shape[1], fmt"Input matrix is not squared.")
  assert(A.sizeof == 4 or A.sizeof == 8, fmt"Wrong data type.")
  var
    A_cp : NdArray[T]  
    B: NdArray[T] = diag(@[1.T].cycle(A.shape[0]))
    blasA_cfloat: BlasMatrix[cfloat]
    blasA_cdouble: BlasMatrix[cdouble]
    ipiv: BlasVector[cint] = @[1.cint].cycle(A.shape[0]).toBlasVector
    ipiv_seq: seq[cint]
    info: cint = 0
    detSign: float = 1.0

  if A.flags.C_Continuous:
    A_cp = A.transpose.copy.transpose
  else:
    A_cp = deepcopy(A)

  if T.sizeof == 4:
    blasA_cfloat = A_cp.toBlasMatrix.forceCast(cfloat)
    sgetrf(blasA_cfloat.m.addr, blasA_cfloat.n.addr, blasA_cfloat.data, blasA_cfloat.lda.addr, ipiv.data, info.addr)
    assert(info==0,fmt"LAPACK return error info {info}")
  else:
    blasA_cdouble = A_cp.toBlasMatrix.forceCast(cdouble)
    dgetrf(blasA_cdouble.m.addr, blasA_cdouble.n.addr, blasA_cdouble.data, blasA_cdouble.lda.addr, ipiv.data, info.addr)
    assert(info==0,fmt"LAPACK return error info {info}")
  
  shallowCopy(ipiv_seq, ipiv.data_buffer)
  for i in 0 ..< A.shape[0]:
    if ipiv_seq[i] != i+1: detSign = -detSign
  result = A_cp.diag.toSeq.prodOfSeq * detSign

proc matrix_rank*[T: SomeFloat](A: NdArray[T], epsilon_multiplier: float = 1e-15): int=
  var
    (U, S, VT) = A.svd
    epsilon = S.data_buffer.max * epsilon_multiplier
  result = 0
  for i in 0 ..< S.data_buffer.len:
    if S.data_buffer[i].abs > epsilon: result += 1

proc inv*[T: SomeFloat](A: NdArray[T]): NdArray[T]=
  # Use xGESV to calculate inverse of a matrix.
  assert(A.shape.len == 2 and A.shape[0] == A.shape[1], fmt"Input matrix {A.shape} is not squared.")
  assert(A.sizeof == 4 or A.sizeof == 8, fmt"Wrong data type.")
  var
    A_cp = A.copy
    B: NdArray[T] = diag(@[1.T].cycle(A.shape[0]))
    blasA_cfloat, blasB_cfloat : BlasMatrix[cfloat]
    blasA_cdouble, blasB_cdouble : BlasMatrix[cdouble]
    ipiv: BlasVector[cint] = @[1.cint].cycle(A.shape[0]).toBlasVector
    info: cint = 0

  if T.sizeof == 4:
    blasA_cfloat = A_cp.toBlasMatrix.forceCast(cfloat)
    blasB_cfloat = B.toBlasMatrix.forceCast(cfloat)
    sgesv(blasA_cfloat.m.addr, blasB_cfloat.m.addr, blasA_cfloat.data, blasA_cfloat.lda.addr, ipiv.data, blasB_cfloat.data, blasB_cfloat.lda.addr, info.addr)
    assert(info==0,fmt"LAPACK return error info {info}")
  else:
    blasA_cdouble = A_cp.toBlasMatrix.forceCast(cdouble)
    blasB_cdouble = B.toBlasMatrix.forceCast(cdouble)
    dgesv(blasA_cdouble.m.addr, blasB_cdouble.m.addr, blasA_cdouble.data, blasA_cdouble.lda.addr, ipiv.data, blasB_cdouble.data, blasB_cdouble.lda.addr, info.addr)
    assert(info==0,fmt"LAPACK return error info {info}")
  
  result = B

proc svd*[T: SomeFloat](A: NdArray[T]): (NdArray[T], NdArray[T], NdArray[T])=
  # Use xGESVD to compute SVD decomposition.
  # return U, S, VT
  assert(A.shape.len == 2, fmt"Input matrix ({A.shape}) is not squared.")
  assert(T.sizeof == 4 or T.sizeof == 8, fmt"Wrong data type.")

  var
    A_cp : NdArray[T]
  
  if A.flags.C_Continuous:
    A_cp = A.transpose.copy.transpose
  else:
    A_cp = deepcopy(A)
  
  var
    blasA = A_cp.toBlasMatrix
    blasA_s: BlasMatrix[cfloat]
    blasA_d: BlasMatrix[cdouble]
    M = blasA.m
    N = blasA.n
    JOBU = "A".cstring
    JOBVT = "A".cstring
    S = newSeq[T](min(M,N).int).toNdArray
    U = newSeq[T]((M*M).int).toNdArray(@[M.int, M.int]).transpose
    VT = newSeq[T]((N*N).int).toNdArray(@[N.int, N.int]).transpose
    blasS_s, blasW_s: BlasVector[cfloat]
    blasS_d, blasW_d: BlasVector[cdouble]
    blasU_s, blasVT_s : BlasMatrix[cfloat]
    blasU_d, blasVT_d : BlasMatrix[cdouble]
    LWORK = max(6 * min(M,N), 4 * min(M,N) + max(M,N)).cint
    WORK = newSeq[T](LWORK.int).toNdArray
    INFO = 0.cint

  blasA.order = CBLAS_ORDER.CblasColMajor
  if T.sizeof == 4:
    blasA_s = blasA.forceCast(cfloat)
    blasS_s = S.toBlasVector.forceCast(cfloat)
    blasU_s = U.toBlasMatrix.forceCast(cfloat)
    blasVT_s = VT.toBlasMatrix.forceCast(cfloat)
    blasW_s = WORK.toBlasVector.forceCast(cfloat)
    blasU_s.order = CBLAS_ORDER.CblasColMajor
    blasVT_s.order = CBLAS_ORDER.CblasColMajor
    sgesvd(JOBU, JOBVT, M.addr, N.addr, blasA_s.data, blasA_s.lda.addr, blasS_s.data, blasU_s.data, blasU_s.lda.addr, blasVT_s.data, blasVT_s.lda.addr, blasW_s.data, LWORK.addr, INFO.addr)
    assert(INFO==0,fmt"LAPACK return error info {INFO}")
  else:
    blasA_d = blasA.forceCast(cdouble)
    blasS_d = S.toBlasVector.forceCast(cdouble)
    blasU_d = U.toBlasMatrix.forceCast(cdouble)
    blasVT_d = VT.toBlasMatrix.forceCast(cdouble)
    blasW_d = WORK.toBlasVector.forceCast(cdouble)
    blasU_d.order = CBLAS_ORDER.CblasColMajor
    blasVT_d.order = CBLAS_ORDER.CblasColMajor
    dgesvd(JOBU, JOBVT, M.addr, N.addr, blasA_d.data, blasA_d.lda.addr, blasS_d.data, blasU_d.data, blasU_d.lda.addr, blasVT_d.data, blasVT_d.lda.addr, blasW_d.data, LWORK.addr, INFO.addr)
    assert(INFO==0,fmt"LAPACK return error info {INFO}")
  result = (U, S, VT)

proc cholesky*[T: SomeFloat](A: NdArray[T]): NdArray[T]=
  assert(A.shape.len == 2 and A.shape[0] == A.shape[1], fmt"Input matrix is not squared.")
  assert(A.sizeof == 4 or A.sizeof == 8, fmt"Wrong data type.")
  {.warning: "I dont check whether the input matrix is symmetric or not.".}
  var
    A_cp: NdArray[T]
  if A.flags.C_Continuous:
    A_cp = A.transpose.copy.transpose
  else:
    A_cp = deepcopy(A)
  
  var
    blasA = A_cp.toBlasMatrix
    blasA_s : BlasMatrix[cfloat]
    blasA_d : BlasMatrix[cdouble]
    n = blasA.m.int
    AP = newSeq[T](n*(n+1) div 2).toNdArray
    blasAP_s : BlasVector[cfloat]
    Uplo = "U".cstring
    INFO = 0.cint
    LDA = n.cint

  if T.sizeof == 4:
    blasA_s = blasA.forceCast(cfloat)
    spotrf(Uplo, blasA_s.m.addr, blasA_s.data, blasA_s.lda.addr, INFO.addr)
    assert(INFO==0,fmt"LAPACK return error info {INFO}")
  else:
    discard 0
  
  for i in 1..<n:
    for j in 0..<i:
      A_cp.set_at(0.T, @[i,j])
  
  result = A_cp.transpose

proc qr*[T: SomeFloat](A: NdArray[T]): (NdArray[T], NdArray[T])=
  assert(A.shape.len == 2, fmt"Input matrix ({A.shape}) is not squared.")
  assert(A.sizeof == 4 or A.sizeof == 8, fmt"Wrong data type.")
  var
    A_cp : NdArray[T]
  
  if A.flags.C_Continuous:
    A_cp = A.transpose.copy.transpose
  else:
    A_cp = deepcopy(A)
  
  var
    blasA = A_cp.toBlasMatrix
    m = blasA.m.int
    n = blasA.n.int
    TAU = newSeq[T](min(m,n)).toNdArray
    WORK = newSeq[T](n).toNdArray
    INFO = 0.cint
    blasA_s : BlasMatrix[cfloat]
    blasT_s, blasW_s : BlasVector[cfloat]
    blasA_d : BlasMatrix[cdouble]
    blasT_d, blasW_d : BlasVector[cdouble]

  blasA.order = CBLAS_ORDER.CblasColMajor

  if T.sizeof == 4:
    blasA_s = blasA.forceCast(cfloat)
    blasT_s = TAU.toBlasVector.forceCast(cfloat)
    blasW_s = WORK.toBlasVector.forceCast(cfloat)
    sgeqr2(blasA.m.cint.addr, blasA.n.cint.addr, blasA_s.data, blasA_s.lda.addr, blasT_s.data, blasW_s.data, INFO.addr)
    assert(INFO==0,fmt"LAPACK return error info {INFO}")
  else:
    blasA_d = blasA.forceCast(cdouble)
    blasT_d = TAU.toBlasVector.forceCast(cdouble)
    blasW_d = WORK.toBlasVector.forceCast(cdouble)
    dgeqr2(blasA.m.cint.addr, blasA.n.cint.addr, blasA_d.data, blasA_d.lda.addr, blasT_d.data, blasW_d.data, INFO.addr)
    assert(INFO==0,fmt"LAPACK return error info {INFO}")

  var
    v: NdArray[T]
    I = @[1.T].cycle(m).diag
    Q = I.copy
    tau: T

  for i in 0 ..< min(m, n):
    tau = TAU.data_buffer[i]
    if tau.abs <= 1e-15: continue
    v = (@[0.T].cycle(m)).toNdArray(@[m, 1])
    v.data_buffer[i] = 1
    for j in i+1 .. m-1:
      v.data_buffer[j] = A_cp.at(j,i)
    Q = Q.dot(I - (v.dot(v.transpose) * tau))

  for i in 0 .. min(m, n)-1:
    for j in 0 .. i-1:
      A_cp.set_at(0.T, @[i,min(m,n)-i-1])
  
  result = (Q, A_cp)

proc eig*[T: SomeFloat](A: NdArray[T]): (NdArray[T], NdArray[T], NdArray[T])=
  # return: 1) vector of eigen value real part; 2) vector of eigen value of imaginary part; 3) right eigen matrix.
  assert(A.shape.len == 2 and A.shape[0] == A.shape[1], fmt"Input matrix is not squared.")
  assert(A.sizeof == 4 or A.sizeof == 8, fmt"Wrong data type.")
  var
    A_cp : NdArray[T]
    transpose_result = true
  
  if A.flags.C_Continuous:
    A_cp = A.transpose.copy.transpose
  else:
    A_cp = deepcopy(A)
  
  var
    JOBVL = "N".cstring
    JOBVR = "V".cstring
    blasA = A_cp.toBlasMatrix
    n = blasA.m.int
    WR = newSeq[T](n).toNdArray
    WI = newSeq[T](n).toNdArray
    VL = newSeq[T](1).toNdArray(@[1,1])
    VR = newSeq[T](n * n).toNdArray(@[n,n])
    LWORK = (5*n).cint
    WORK = newSeq[T](LWORK.int).toNdArray
    INFO = 0.cint
    blasWR_s, blasWI_s, blasW_s: BlasVector[cfloat]
    blasWR_d, blasWI_d, blasW_d : BlasVector[cdouble]
    blasA_s, blasVL_s, blasVR_s : BlasMatrix[cfloat]
    blasA_d, blasVL_d, blasVR_d: BlasMatrix[cdouble]
  
  if T.sizeof == 4:
    blasA_s = blasA.forceCast(cfloat)
    blasWR_s = WR.toBlasVector.forceCast(cfloat)
    blasWI_s = WI.toBlasVector.forceCast(cfloat)
    blasVR_s = VR.toBlasMatrix.forceCast(cfloat)
    blasVL_s = VL.toBlasMatrix.forceCast(cfloat)
    blasW_s = WORK.toBlasVector.forceCast(cfloat)

    sgeev(JOBVL, JOBVR, blasA.m.addr, blasA_s.data, blasA.lda.addr, blasWR_s.data, blasWI_s.data, blasVL_s.data, blasVL_s.lda.addr, blasVR_s.data, blasVR_s.lda.addr, blasW_s.data, LWORK.addr, INFO.addr)
    assert(INFO==0,fmt"LAPACK return error info {INFO}")
  else:
    blasA_d = blasA.forceCast(cdouble)
    blasWR_d = WR.toBlasVector.forceCast(cdouble)
    blasWI_d = WI.toBlasVector.forceCast(cdouble)
    blasVR_d = VR.toBlasMatrix.forceCast(cdouble)
    blasVL_d = VL.toBlasMatrix.forceCast(cdouble)
    blasW_d = WORK.toBlasVector.forceCast(cdouble)

    dgeev(JOBVL, JOBVR, blasA.m.addr, blasA_d.data, blasA.lda.addr, blasWR_d.data, blasWI_d.data, blasVL_d.data, blasVL_d.lda.addr, blasVR_d.data, blasVR_d.lda.addr, blasW_d.data, LWORK.addr, INFO.addr)
    assert(INFO==0,fmt"LAPACK return error info {INFO}")

  result = (WR, WI, VR.transpose)

proc eigvals*[T: SomeFloat](A: NdArray[T]): (NdArray[T], NdArray[T])=
  # return: 1) vector of eigen value real part, 2) vector of eigen value imaginary part
  var
    (WR, WI, V) = A.eig
  result = (WR, WI)

when isMainModule:
  import ../ndarray/random_ndarray
  import ../ndarray/special_ndarray
  var
    a = normal(@[30,1])
    (u,s,vt) = a.svd
    S = zeros(@[30,1])
  S.set_at(s.at(0), 0,0)
  # echo a
  echo u.dot(S).dot(vt) - a
  # echo a.eigvals
  # echo a.cholesky
  
  # echo a.set_at(0,1)
  # echo a.transpose[0..1,_].det
  # echo a[_,0..3]
  # echo a[_, 0..3].inv
  # echo a.shape.len
  # echo a
  # echo "U".cstring

  # var
  #   m = 10
  #   x = (@[0.cfloat].cycle(m))
  # echo x