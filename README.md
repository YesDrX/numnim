# numnim
Numpy like ndarray and dataframe library for nim-lang.

# DEPENDENCIES
BLAS
LAPACK
nimblas
nimlapack

# IMPORT
import ./src/numnim

# EXAMPLES

## 1. create a 2d-ndarray, and run svd decomposition

```nim
  import ./src/numnim #suppose your nim file is in numnim folder.
  
  when isMainModule:
    var
      a = arange(9.0).reshape(@[3,3])
      (u,s,vt) = a.svd
     echo u
     echo s
     echo vt
```
### results
```nim
[
    [0.0, 1.0, 2.0],
    [3.0, 4.0, 5.0],
    [6.0, 7.0, 8.0]
], shape = [3, 3]
[
    [-0.1351189484190462, -0.4963351427771625, -0.8575513371352792],
    [0.9028157082768692, 0.2949317876691269, -0.3129521329386146],
    [0.4082482904638625, -0.8164965809277261, 0.4082482904638632]
], shape = [3, 3]
[14.22670739082269, 1.265225994006975, 5.89938022126432e-16], shape = [3]
[
    [-0.4663281017098022, -0.7847747669803522, -0.4082482904638625],
    [-0.5709907891066073, -0.08545672641019784, 0.816496580927726],
    [-0.6756534765034119, 0.6138613141599545, -0.4082482904638636]
], shape = [3, 3]
```

## 2. matrix determinant and inverse
```nim
  var
    a = normal(@[3,3])
  echo a
  echo a.det
  echo a.inv
```
### results
```nim
[
    [-0.7641806091526053, 0.691080778722737, 0.4890344984537413],
    [-0.1941453136646491, 2.355007647679016, 2.229789109753814],
    [0.3808236427287697, -0.2836523127918563, -1.556891991219836]
], shape = [3, 3]
2.284821236009206
[
    [-1.327897193923375, 0.4101950508806507, 0.1703784980564418],
    [0.239359263077958, 0.439207564818849, 0.7042204522261785],
    [-0.3684195553411621, 0.02031594492586134, -0.7289327751166562]
], shape = [3, 3]
```

## 3. slice
```nim
  var
    a = normal(@[3,3,3])
  echo a
  echo a[_, 0, 0]
  echo a[0 .. 1,0 .. -2, 0]
```
### results
```nim
[
    [
        [1.36193288377643, 1.867257466610367, -0.8804896138694298],
        [-0.5344388992710561, -1.069513851548713, -0.5394889064410189],
        [-0.8632317038782776, 0.3756828981732284, 0.8703486148969456]
    ],
    [
        [0.4042109798946042, -1.294245937262206, -0.829936586472778],
        [0.5035422285020081, 1.883046119540039, -0.1516047931302702],
        [-0.7190116889214877, 0.7589246642340399, -0.4370144409793927]
    ],
    [
        [-2.475790532123788, 1.138078855623718, 0.3861783221141945],
        [2.037813453536978, 0.2993076770207514, -1.535841081399497],
        [0.8382056498275173, -0.5162988860661161, 1.233868332332112]
    ]
], shape = [3, 3, 3]
[
    [
        [1.36193288377643]
    ],
    [
        [0.4042109798946042]
    ],
    [
        [-2.475790532123788]
    ]
], shape = [3, 1, 1], memory is not contiguous (maybe a view).
[
    [
        [1.36193288377643],
        [-0.5344388992710561]
    ],
    [
        [0.4042109798946042],
        [0.5035422285020081]
    ]
], shape = [2, 2, 1], memory is not contiguous (maybe a view).
```

## 4. math operations
```nim
  var
    a = normal(@[2,3])
    b = normal(@[3,2])
  echo a
  echo b
  echo a.dot(b)
  echo a * b.transpose
  echo a + b.transpose
```
### results
```nim
[
    [0.2372791735118804, -1.745413473067925, -0.8839529907804951],
    [-1.150742885872321, 1.19963350960213, -0.2443910386957094]
], shape = [2, 3]
[
    [-0.7629467805985998, -0.9473116014982472],
    [-0.4176046315682378, 0.9960004291787755],
    [0.9197626745251068, 0.2338734043498078]
], shape = [3, 2]
[
    [-0.2651655981339519, -2.169942977370819],
    [0.1522013148979854, 2.227791012337034]
], shape = [2, 2]
[
    [-0.1810313815339857, 0.7288927503547691, -0.8130269669547352],
    [1.090112086128423, 1.194835490420962, -0.05715656421235117]
], shape = [2, 3]
[
    [-0.5256676070867194, -2.163018104636163, 0.0358096837446118],
    [-2.098054487370568, 2.195633938780905, -0.01051763434590156]
], shape = [2, 3]
```

## 5. construct ndarray from sequence/array
```nim
  import sequtils
  var
    a = @[1.0].cycle(10).toNdArray(@[5,2])
    b = @[1.0].cycle(3).diag
    c = xrange(3.0).diag
  echo a
  echo b
  echo c
```
### results
```nim
[
    [1.0, 1.0],
    [1.0, 1.0],
    [1.0, 1.0],
    [1.0, 1.0],
    [1.0, 1.0]
], shape = [5, 2]
[
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
], shape = [3, 3]
[
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 2.0]
], shape = [3, 3]
```

## 6. assign values
```nim
  import sequtils
  import strformat
  var
    a = @[1.0].cycle(3).diag
    b = @[2.0].cycle(4).toNdArray(@[2,2])
  echo a
  echo b
  a[0..1, 0..1] = b
  echo a
```
### results
```
[
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
], shape = [3, 3]
[
    [2.0, 2.0],
    [2.0, 2.0]
], shape = [2, 2]
[
    [2.0, 2.0, 0.0],
    [2.0, 2.0, 0.0],
    [0.0, 0.0, 1.0]
], shape = [3, 3]

```

# WHAT'S NEXT
. add some tests

. add some docs

. do some debuging

. add math functions (basically map some function to the underlying data_buffer: NdArray[T].data_buffer)

. add some algorithm, e.g. sorting for ndarray and dataframe

. make it nimble package

# WHY NUMNIM?
I like python, and apis for projects out there are not intuitive to me.

# NO WORRANTY
I wrote this project over the weekend, and there are supposed to be many bugs. Also, I'm not a software engineer. 

# CONTRIBUTE
Yes, please. When I have time, I'll layout the contribution guidence. If you can join us, please avoid template/macro as much as you can.

