[![nimble](https://raw.githubusercontent.com/yglukhov/nimble-tag/master/nimble.png)](https://github.com/yglukhov/nimble-tag)

[![Linux](https://img.shields.io/travis/YesDrX/numnim/master.svg?label=Linux%20Install%20and%20Test)](https://travis-ci.org/YesDrX/numnim)
[![MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# What is numnim?

Numpy like ndarray and dataframe library for nim-lang.

# Dependencies

BLAS (to compile and install: [BLAS](https://github.com/xianyi/OpenBLAS))

LAPACK (to compile and install: [LAPACK](http://www.netlib.org/lapack/))

nimblas

nimlapack

*Check [.travis.yml](https://github.com/YesDrX/numnim/blob/master/.travis.yml)* for detailed setup on linux.


# Installation
```
git clone https://github.com/YesDrX/numnim.git
cd numnim
nimble install
```
or
```
nimble install numnim
```

# Testing
```nim
nimble test numnim
```

# Documentation

At this moment, nothing is stable. So documentation is still at its minimum level. Check wiki for some basic descriptions : 

[Wiki](https://github.com/YesDrX/numnim/wiki)

[Create ndarray](https://github.com/YesDrX/numnim/wiki/1.-Construct-NdArray)

[Slice ndarray](https://github.com/YesDrX/numnim/wiki/2.-Slice-NdArray)

[DataFrame](https://github.com/YesDrX/numnim/wiki/3.-DataFrame)


# Examples

```nim
import numnim

# initialize a matrix/ndarray
var
  A = @[5,3,3,4].astype(float).toNdarray(@[2,2])

echo "A = \n" & $A & "\n"
# A = 
# [
#     [5.0, 3.0],
#     [3.0, 4.0]
# ], shape = [2, 2]

##################################################
# Matrix Inverse
##################################################
echo "A.inv = \n" & $(A.inv) & "\n"
# A.inv = 
# [
#     [0.3636363636363636, -0.2727272727272728],
#     [-0.2727272727272728, 0.4545454545454546]
# ], shape = [2, 2]

##################################################
# Matrix Determinant
##################################################
echo "A.det = " & $(A.det) & "\n"
# A.det = 11.0

##################################################
# Matrix Eigenvalues
##################################################
echo "A.eigvals = " & $(A.eigvals) & "\n"
# A.eigvals = ([7.54138126514911, 1.45861873485089], shape = [2], [0.0, 0.0], shape = [2])

##################################################
# Matrix Cholesky Decomposition
##################################################
echo "A.cholesky = \n" & $(A.cholesky) & "\n" # Matrix is assumed symmetric and positive definite.
# A.cholesky = 
# [
#     [2.23606797749979, 0.0],
#     [1.341640786499874, 1.483239697419133]
# ], shape = [2, 2]

echo "A.cholesky.dot(A.cholesky.transpose) = \n" & $(A.cholesky.dot(A.cholesky.transpose)) & "\n"
# A.cholesky.dot(A.cholesky.transpose) = 
# [
#     [5.000000000000001, 3.0],
#     [3.0, 4.0]
# ], shape = [2, 2], memory is F_Continuous.

##################################################
# Matrix QR Decomposition
##################################################
echo "A.qr = \n" & $(A.qr) & "\n"
# A.qr = 
# ([
#     [-0.8574929257125441, -0.5144957554275266],
#     [-0.5144957554275266, 0.8574929257125441]
# ], shape = [2, 2], memory is F_Continuous., [
#     [-5.8309518948453, -4.630461798847739],
#     [0.0, 1.886484436567597]
# ], shape = [2, 2], memory is F_Continuous.)

echo "A.qr[0].dot(A.qr[1]) = \n" & $(A.qr[0].dot(A.qr[1])) & "\n"
# A.qr[0].dot(A.qr[1]) = 
# [
#     [4.999999999999999, 3.0],
#     [3.0, 4.0]
# ], shape = [2, 2], memory is F_Continuous.

##################################################
# Matrix Rank
##################################################
echo "A.matrix_rank = " & $(A.matrix_rank) & "\n"
# A.matrix_rank = 2

##################################################
# Matrix Transpose
##################################################
echo "A.transpose = " & $(A.transpose) & "\n"
# A.transpose = [
#     [5.0, 3.0],
#     [3.0, 4.0]
# ], shape = [2, 2], memory is F_Continuous.

##################################################
# NdArray Reshape
##################################################
echo "A.reshape(@[4]) = " & $(A.reshape(@[4])) & "\n"
# A.reshape(@[4]) = [5.0, 3.0, 3.0, 4.0], shape = [4]

##################################################
# NdArray Operators
##################################################
echo "A * 2.0 = \n" & $(A * 2.0) & "\n"
# A * 2.0 = 
# [
#     [10.0, 6.0],
#     [6.0, 8.0]
# ], shape = [2, 2]

echo "A + A = \n" & $(A + A) & "\n"
# A + A = 
# [
#     [10.0, 6.0],
#     [6.0, 8.0]
# ], shape = [2, 2]

echo "A - 2.0 * A = \n" & $(A - 2.0 * A) & "\n"
# A - 2.0 * A = 
# [
#     [-5.0, -3.0],
#     [-3.0, -4.0]
# ], shape = [2, 2]

echo "A / A = \n" & $(A / A) & "\n"
# A / A = 
# [
#     [1.0, 1.0],
#     [1.0, 1.0]
# ], shape = [2, 2]

echo "A.dot(A) = \n" & $(A.dot(A)) & "\n"
# A.dot(A) = 
# [
#     [34.0, 27.0],
#     [27.0, 25.0]
# ], shape = [2, 2], memory is F_Continuous.

##################################################
# NdArray Slice and Assign
##################################################
echo "A[0,0 .. -1] = " & $(A[0,0 .. -1]) & "\n"
# A[0,0 .. -1] = [
#     [5.0, 3.0]
# ], shape = [1, 2]

A[0,0 .. -1] = @[0.0,0.0].toNdArray(@[1,2])
echo A
# [
#     [0.0, 0.0],
#     [3.0, 4.0]
# ], shape = [2, 2]
```

# Why numnim?
There are multiple nim based ndarray projectes, such as Arraymancer with a focus on ML, and neo. Both are great, and their support for GPU backend is a great inspiration for my next steps. However, I'm just new to nim and more familiar with numpy's api. So I decided to create some new wheel for nim with more intuitive interface.

One can easily use nimy to interop with numpy in python, but because of the GIL in python, you can't easily do parallel in nim with nimpy. With something written in nim, one can easily run true threading in nim.
