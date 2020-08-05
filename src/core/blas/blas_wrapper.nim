import nimblas/cblas
import blas_vector
import blas_matrix
import ../common


############################################################################
# level 1
############################################################################
#?DOT
proc dot*(x : BlasVector[cfloat], y:BlasVector[cfloat], alpha: float32|float|cfloat|cdouble): float32 =
  result = sdsdot(x.size, alpha.cfloat, x.data, x.stride, y.data, y.stride).float32

proc dot*(x : BlasVector[cfloat], y:BlasVector[cfloat]): float =
  result = dsdot(x.size, x.data, x.stride, y.data, y.stride).float

proc dot*(x : BlasVector[cdouble], y : BlasVector[cdouble]): float =
  result = ddot(x.size, x.data, x.stride, y.data, y.stride).float


#?NRM2
proc nrm2*(x : BlasVector[cfloat]) : float32 =
  # return L2_Norm(x)
  result = snrm2(x.size, x.data, x.stride).float32

proc nrm2*(x : BlasVector[cdouble]) : float =
  # return L2_Norm(x)
  result = dnrm2(x.size, x.data, x.stride).float

#I?AMAX
proc Iamax*(x : BlasVector[cfloat]) : int =
  result = isamax(x.size, x.data, x.stride).int


proc Iamax*(x : BlasVector[cdouble]) : int =
  result = idamax(x.size, x.data, x.stride).int


#?SWAP
proc swap*(x : BlasVector[cfloat], y : BlasVector[cfloat]) =
  sswap(x.size, x.data, x.stride, y.data, y.stride)

proc swap*(x : BlasVector[cdouble], y : BlasVector[cdouble]) =
  dswap(x.size, x.data, x.stride, y.data, y.stride)

#?AXPY
proc axpy*(x : BlasVector[cfloat], y : BlasVector[cfloat], alpha: float32|float|cfloat|cdouble) =
  # y = alpha * x + y
  saxpy(x.size, alpha.cfloat, x.data, x.stride, y.data, y.stride)

proc axpy*(x : BlasVector[cdouble], y : BlasVector[cdouble], alpha: float32|float|cfloat|cdouble)  =
  # y = alpha * x + y
  daxpy(x.size, alpha.cdouble, x.data, x.stride, y.data, y.stride)

#?COPY
proc copy*(x : BlasVector[cfloat], y : BlasVector[cfloat]) =
  scopy(x.size, x.data, x.stride, y.data, y.stride)

proc copy*(x : BlasVector[cdouble], y : BlasVector[cdouble]) =
  dcopy(x.size, x.data, x.stride, y.data, y.stride)


#?ROTG
proc rotg*(a: ptr cfloat, b: ptr cfloat, c: ptr cfloat, s: ptr cfloat) =
  srotg(a, b, c, s)
proc rotg*(a: ptr cdouble, b: ptr cdouble, c: ptr cdouble, s: ptr cdouble) =
  drotg(a, b, c, s)

#?ROTMG
proc rotmg*(d1: ptr cfloat, d2: ptr cfloat,  b1: ptr cfloat, b2: float32|float|cfloat|cdouble, P: ptr cfloat) =
  srotmg(d1, d2, b1, b2.cfloat, P)

proc rotmg*(d1: ptr cdouble, d2: ptr cdouble,  b1: ptr cdouble, b2: float32|float|cfloat|cdouble, P: ptr cdouble) =
  drotmg(d1, d2, b1, b2.cdouble, P)

#?ROT
proc rot*(x: BlasVector[cfloat], y : BlasVector[cfloat], c: float32|float|cfloat|cdouble, s: float32|float|cfloat|cdouble) =
  srot(x.size, x.data, x.stride, y.data, y.stride, c.cfloat, s.cfloat)

proc rot*(x: BlasVector[cdouble], y : BlasVector[cdouble], c: float32|float|cfloat|cdouble, s: float32|float|cfloat|cdouble) =
  drot(x.size, x.data, x.stride, y.data, y.stride, c.cdouble, s.cdouble)

#?ROTM
proc rotm*(x: BlasVector[cfloat], y : BlasVector[cfloat], P : ptr cfloat) =
  srotm(x.size, x.data, x.stride, y.data, y.stride, P)

proc rotm*(x: BlasVector[cdouble], y : BlasVector[cdouble], P : ptr cdouble) =
  drotm(x.size, x.data, x.stride, y.data, y.stride, P)

#?SCAL
proc scal*(x: BlasVector[cfloat], alpha: float32|float|cfloat|cdouble) = 
  # x = alpha * x
  sscal(x.size, alpha.cfloat, x.data, x.stride)

proc scal*(x: BlasVector[cdouble], alpha: float32|float|cfloat|cdouble) = 
  # x = alpha * x
  dscal(x.size, alpha.cdouble, x.data, x.stride)


############################################################################
# level 2
############################################################################

#?GEMV
proc gemv*(TransA: CBLAS_TRANSPOSE, alpha: float32|float|cfloat|cdouble, beta: float32|float|cfloat|cdouble, A: BlasMatrix[cfloat], x: BlasVector[cfloat], y: BlasVector[cfloat]) =
  # y = alpha * A * x + beta * y, A( or A.T, A.H)
  sgemv(A.order, TransA, A.m, A.n, alpha.cfloat, A.data, A.lda, x.data, x.stride, beta.cfloat, y.data, y.stride)

proc gemv*(TransA: CBLAS_TRANSPOSE, alpha: float32|float|cfloat|cdouble, beta: float32|float|cfloat|cdouble, A: BlasMatrix[cdouble], x: BlasVector[cdouble], y: BlasVector[cdouble]) =
  # y = alpha * A * x + beta * y, A( or A.T, A.H)
  dgemv(A.order, TransA, A.m, A.n, alpha.cdouble, A.data, A.lda, x.data, x.stride, beta.cdouble, y.data, y.stride)

#?GBMV
proc gbmv*(TransA: CBLAS_TRANSPOSE, KL: int, KU: int, alpha: float32|float|cfloat|cdouble, beta: float32|float|cfloat|cdouble, A: BlasMatrix[cfloat], x: BlasVector[cfloat], y: BlasVector[cfloat]) =
  # y = alpha * A * x + beta * y, A( or A.T, A.H)
  sgbmv(A.order, TransA, A.m, A.n, KL.cint, KU.cint, alpha.cfloat, A.data, A.lda, x.data, x.stride, beta.cfloat, y.data, y.stride)

proc gbmv*(TransA: CBLAS_TRANSPOSE, KL: int, KU: int, alpha: float32|float|cfloat|cdouble, beta: float32|float|cfloat|cdouble, A: BlasMatrix[cdouble], x: BlasVector[cdouble], y: BlasVector[cdouble]) =
  # y = alpha * A * x + beta * y, A( or A.T, A.H)
  dgbmv(A.order, TransA, A.m, A.n, KL.cint, KU.cint, alpha.cdouble, A.data, A.lda, x.data, x.stride, beta.cdouble, y.data, y.stride)

#?TRMV
proc trmv*(Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, A: BlasMatrix[cfloat], x: BlasVector[cfloat]) =
  strmv(A.order, Uplo, TransA, Diag, A.m, A.data, A.lda, x.data, x.stride)

proc trmv*(Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, A: BlasMatrix[cdouble], x: BlasVector[cdouble]) =
  dtrmv(A.order, Uplo, TransA, Diag, A.m, A.data, A.lda, x.data, x.stride)

#?TBMV
proc tbmv*(Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, A: BlasMatrix[cfloat], K: int, x: BlasVector[cfloat]) =
  stbmv(A.order, Uplo , TransA, Diag, A.m, K.cint, A.data, A.lda, x.data, x.stride  )

proc tbmv*(Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, A: BlasMatrix[cdouble], K: int, x: BlasVector[cdouble]) =
  dtbmv(A.order, Uplo , TransA, Diag, A.m, K.cint, A.data, A.lda, x.data, x.stride  )

#?TPMV
proc tpmv*(Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, A: BlasMatrix[cfloat], x: BlasVector[cfloat]) =
  stpmv(A.order, Uplo , TransA, Diag, A.m, A.data, x.data, x.stride )

proc tpmv*(Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, A: BlasMatrix[cdouble], x: BlasVector[cdouble]) =
  dtpmv(A.order, Uplo , TransA, Diag, A.m, A.data, x.data, x.stride )

#?TRSV
proc trsv*(Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, A: BlasMatrix[cfloat], x: BlasVector[cfloat]) =
  strsv(A.order, Uplo , TransA, Diag, A.m, A.data, A.lda, x.data, x.stride)

proc trsv*(Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, A: BlasMatrix[cdouble], x: BlasVector[cdouble]) =
  dtrsv(A.order, Uplo , TransA, Diag, A.m, A.data, A.lda, x.data, x.stride)

#?TBSV
proc tbsv*(Uplo : CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, A: BlasMatrix[cfloat], K: int, x: BlasVector[cfloat]) =
  stbsv(A.order, Uplo, TransA, Diag, A.m, K.cint, A.data, A.lda, x.data, x.stride)

proc tbsv*(Uplo : CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, A: BlasMatrix[cdouble], K: int, x: BlasVector[cdouble]) =
  dtbsv(A.order, Uplo, TransA, Diag, A.m, K.cint, A.data, A.lda, x.data, x.stride)

#?TPSV
proc tpsv*(Uplo : CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, A: BlasMatrix[cfloat], x: BlasVector[cfloat]) =
  stpsv(A.order, Uplo , TransA, Diag, A.m, A.data, x.data, x.stride)

proc tpsv*(Uplo : CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, A: BlasMatrix[cdouble], x: BlasVector[cdouble]) =
  dtpsv(A.order, Uplo , TransA, Diag, A.m, A.data, x.data, x.stride)

#?SYMV
proc symv*(Uplo: CBLAS_UPLO, alpha: float32|float|cfloat|cdouble, beta: float32|float|cfloat|cdouble, A: BlasMatrix[cfloat], x: BlasVector[cfloat], y: BlasVector[cfloat]) =
  ssymv(A.order, Uplo, A.m, alpha.cfloat, A.data, A.lda, x.data, x.stride, beta.cfloat, y.data, y.stride)

proc symv*(Uplo: CBLAS_UPLO, alpha: float32|float|cfloat|cdouble, beta: float32|float|cfloat|cdouble, A: BlasMatrix[cdouble], x: BlasVector[cdouble], y: BlasVector[cdouble]) =
  dsymv(A.order, Uplo, A.m, alpha.cdouble, A.data, A.lda, x.data, x.stride, beta.cdouble, y.data, y.stride)

#?SBMV
proc sbmv*(Uplo: CBLAS_UPLO, alpha: float32|float|cfloat|cdouble, K:int, beta: float32|float|cfloat|cdouble, A: BlasMatrix[cfloat], x: BlasVector[cfloat], y: BlasVector[cfloat]) =
  ssbmv(A.order, Uplo, A.m, K.cint, alpha.cfloat, A.data, A.lda, x.data, x.stride, beta.cfloat, y.data, y.stride)

proc sbmv*(Uplo: CBLAS_UPLO, alpha: float32|float|cfloat|cdouble, K:int, beta: float32|float|cfloat|cdouble, A: BlasMatrix[cdouble], x: BlasVector[cdouble], y: BlasVector[cdouble]) =
  dsbmv(A.order, Uplo, A.m, K.cint, alpha.cdouble, A.data, A.lda, x.data, x.stride, beta.cdouble, y.data, y.stride)

#?SPMV
proc spmv*(Uplo: CBLAS_UPLO, alpha: float32|float|cfloat|cdouble, beta: float32|float|cfloat|cdouble, A: BlasMatrix[cfloat], x: BlasVector[cfloat], y: BlasVector[cfloat]) =
  sspmv(A.order, Uplo, A.m, alpha.cfloat, A.data, x.data, x.stride, beta.cfloat, y.data, y.stride)

proc spmv*(Uplo: CBLAS_UPLO, alpha: float32|float|cfloat|cdouble, beta: float32|float|cfloat|cdouble, A: BlasMatrix[cdouble], x: BlasVector[cdouble], y: BlasVector[cdouble]) =
  dspmv(A.order, Uplo, A.m, alpha.cdouble, A.data, x.data, x.stride, beta.cdouble, y.data, y.stride)

#?GER
proc ger*(alpha: float32|float|cfloat|cdouble, A: BlasMatrix[cfloat], x: BlasVector[cfloat], y: BlasVector[cfloat]) =
  sger(A.order, A.m, A.n, alpha.cfloat, x.data, x.stride, y.data, y.stride, A.data, A.lda)

proc ger*(alpha: float32|float|cfloat|cdouble, A: BlasMatrix[cdouble], x: BlasVector[cdouble], y: BlasVector[cdouble]) =
  dger(A.order, A.m, A.n, alpha.cdouble, x.data, x.stride, y.data, y.stride, A.data, A.lda)

#?SYR
proc syr*(Uplo: CBLAS_UPLO, alpha: float32|float|cfloat|cdouble, A: BlasMatrix[cfloat], x: BlasVector[cfloat]) =
  ssyr(A.order, Uplo, A.m, alpha.cfloat, x.data, x.stride, A.data, A.lda)

proc syr*(Uplo: CBLAS_UPLO, alpha: float32|float|cfloat|cdouble, A: BlasMatrix[cdouble], x: BlasVector[cdouble]) =
  dsyr(A.order, Uplo, A.m, alpha.cdouble, x.data, x.stride, A.data, A.lda)

#?SPR
proc spr*(Uplo: CBLAS_UPLO, alpha: float32|float|cfloat|cdouble, A: BlasMatrix[cfloat], x: BlasVector[cfloat]) =
  sspr(A.order, Uplo, A.m, alpha.cfloat, x.data, x.stride, A.data)

proc spr*(Uplo: CBLAS_UPLO, alpha: float32|float|cfloat|cdouble, A: BlasMatrix[cdouble], x: BlasVector[cdouble]) =
  dspr(A.order, Uplo, A.m, alpha.cfloat, x.data, x.stride, A.data)

#?SYR2
proc syr2*(Uplo: CBLAS_UPLO, alpha: float32|float|cfloat|cdouble, A: BlasMatrix[cfloat], x: BlasVector[cfloat], y: BlasVector[cfloat]) =
  ssyr2(A.order, Uplo, A.m, alpha.cfloat, x.data, x.stride, y.data, y.stride, A.data, A.lda)

proc syr2*(Uplo: CBLAS_UPLO, alpha: float32|float|cfloat|cdouble, A: BlasMatrix[cdouble], x: BlasVector[cdouble], y: BlasVector[cdouble]) =
  dsyr2(A.order, Uplo, A.m, alpha.cdouble, x.data, x.stride, y.data, y.stride, A.data, A.lda)

#?SPR2
proc spr2*(Uplo: CBLAS_UPLO, alpha: float32|float|cfloat|cdouble, A: BlasMatrix[cfloat], x: BlasVector[cfloat], y: BlasVector[cfloat]) =
  sspr2(A.order, Uplo, A.m, alpha.cfloat, x.data, x.stride, y.data, y.stride, A.data)

proc spr2*(Uplo: CBLAS_UPLO, alpha: float32|float|cfloat|cdouble, A: BlasMatrix[cdouble], x: BlasVector[cdouble], y: BlasVector[cdouble]) =
  dspr2(A.order, Uplo, A.m, alpha.cdouble, x.data, x.stride, y.data, y.stride, A.data)


############################################################################
# level 3
############################################################################

#?GEMM
proc gemm*(TransA: CBLAS_TRANSPOSE, TransB: CBLAS_TRANSPOSE, alpha: float32|float|cfloat|cdouble, A: BlasMatrix[cfloat], B: BlasMatrix[cfloat], C: BlasMatrix[cfloat], beta: float32|float|cfloat|cdouble) =
  sgemm(A.order, TransA, TransB, C.m, C.n, A.n, alpha.cfloat, A.data, A.lda, B.data, B.lda, beta.cfloat, C.data, C.lda)

proc gemm*(TransA: CBLAS_TRANSPOSE, TransB: CBLAS_TRANSPOSE, alpha: float32|float|cfloat|cdouble, A: BlasMatrix[cdouble], B: BlasMatrix[cdouble], C: BlasMatrix[cdouble], beta: float32|float|cfloat|cdouble) =
  dgemm(A.order, TransA, TransB, C.m, C.n, A.n, alpha.cdouble, A.data, A.lda, B.data, B.lda, beta.cdouble, C.data, C.lda)

#?SYMM
proc symm*(Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, alpha: float32|float|cfloat|cdouble, beta: float32|float|cfloat|cdouble, A: BlasMatrix[cfloat], B: BlasMatrix[cfloat], C: BlasMatrix[cfloat]) =
  ssymm(A.order, Side, Uplo, C.m, C.n, alpha.cfloat, A.data, A.lda, B.data, B.lda, beta.cfloat, C.data, C.lda)

proc symm(Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, alpha: float32|float|cfloat|cdouble, beta: float32|float|cfloat|cdouble, A: BlasMatrix[cdouble], B: BlasMatrix[cdouble], C: BlasMatrix[cdouble]) =
  dsymm(A.order, Side, Uplo, C.m, C.n, alpha.cdouble, A.data, A.lda, B.data, B.lda, beta.cdouble, C.data, C.lda)  

#?SYRK
proc syrk*(Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, alpha: float32|float|cfloat|cdouble, beta: float32|float|cfloat|cdouble, A: BlasMatrix[cfloat], C: BlasMatrix[cfloat]) = 
  ssyrk(A.order, Uplo, Trans, C.m, A.n, alpha.cfloat, A.data, A.lda, beta.cfloat, C.data, C.lda)

proc syrk*(Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, alpha: float32|float|cfloat|cdouble, beta: float32|float|cfloat|cdouble, A: BlasMatrix[cdouble], C: BlasMatrix[cdouble]) = 
  dsyrk(A.order, Uplo, Trans, C.m, A.n, alpha.cdouble, A.data, A.lda, beta.cdouble, C.data, C.lda)

#?SYRK2k
proc syrk2k*(Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, alpha: float32|float|cfloat|cdouble, beta: float32|float|cfloat|cdouble, A: BlasMatrix[cfloat], B: BlasMatrix[cfloat], C: BlasMatrix[cfloat]) =
  ssyr2k(A.order, Uplo, Trans, C.m, A.n, alpha.cfloat, A.data, A.lda, B.data, B.lda, beta.cfloat, C.data, C.lda)

proc syrk2k*(Uplo: CBLAS_UPLO, Trans: CBLAS_TRANSPOSE, alpha: float32|float|cfloat|cdouble, beta: float32|float|cfloat|cdouble, A: BlasMatrix[cdouble], B: BlasMatrix[cdouble], C: BlasMatrix[cdouble]) =
  dsyr2k(A.order, Uplo, Trans, C.m, A.n, alpha.cdouble, A.data, A.lda, B.data, B.lda, beta.cdouble, C.data, C.lda)

#?TRMM
proc trmm*(Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, alpha: float32|float|cfloat|cdouble, A: BlasMatrix[cfloat], B: BlasMatrix[cfloat]) =
  strmm(A.order, Side, Uplo, TransA, Diag, B.m, B.n, alpha.cfloat, A.data, A.lda, B.data, B.lda)

proc trmm*(Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, alpha: float32|float|cfloat|cdouble, A: BlasMatrix[cdouble], B: BlasMatrix[cdouble]) =
  dtrmm(A.order, Side, Uplo, TransA, Diag, B.m, B.n, alpha.cdouble, A.data, A.lda, B.data, B.lda)

#?TRSM
proc trsm*(Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, alpha: float32|float|cfloat|cdouble, A: BlasMatrix[cfloat], B: BlasMatrix[cfloat]) =
  strsm(A.order, Side, Uplo, TransA, Diag, B.m, B.n, alpha.cfloat, A.data, A.lda, B.data, B.lda)

proc trsm*(Side: CBLAS_SIDE, Uplo: CBLAS_UPLO, TransA: CBLAS_TRANSPOSE, Diag: CBLAS_DIAG, alpha: float32|float|cfloat|cdouble, A: BlasMatrix[cdouble], B: BlasMatrix[cdouble]) =
  dtrsm(A.order, Side, Uplo, TransA, Diag, B.m, B.n, alpha.cdouble, A.data, A.lda, B.data, B.lda)

