import sugar
import nimpy
import sequtils
import ../src/numnim

let
  np* = pyImport("numpy")
  py* = pyBuiltinsModule()

proc typename*(obj: PyObject): string=
  return  py.type(obj).getAttr("__name__").to(string)

proc run() =

  ####################
  # Test inv
  for i_test in 0 .. 10:
    var
      A = np.random.normal(0,1,@[100,100])
      B = A.ravel().tolist().to(seq[float]).toNdArray(A.shape.to(seq[int]))
      invA = np.linalg.inv(A).ravel().tolist().to(seq[float]).toNdArray(A.shape.to(seq[int]))
      invB = B.inv
      max_diff = (invA - invB).data_buffer.map(x=>x.abs).max
    echo "Test " & $i_test & " max diff about inv (compare with numpy): \nmax_diff=" & $max_diff
    doAssert(max_diff < 1e-10)
    echo " ... good."
  
  for i_test in 0 .. 10:
    var
      A = np.random.normal(0,1,@[1,1])
      B = A.ravel().tolist().to(seq[float]).toNdArray(A.shape.to(seq[int]))
      invA = np.linalg.inv(A).ravel().tolist().to(seq[float]).toNdArray(A.shape.to(seq[int]))
      invB = B.inv
      max_diff = (invA - invB).data_buffer.map(x=>x.abs).max
    echo "Test " & $i_test & " max diff about inv (compare with numpy): \nmax_diff=" & $max_diff
    doAssert(max_diff < 1e-10)
    echo " ... good."

  ####################
  # Test svd
  for i_test in 0 .. 10:
    var
      A = np.random.normal(0,1,@[30,50])
      B = A.ravel().tolist().to(seq[float]).toNdArray(A.shape.to(seq[int]))
      (u, s, vt) = B.svd
      S = newSeq[float](B.data_buffer.len).toNdArray(B.shape)
    for i in 0..<min(B.shape):
      S.set_at(s.at(i),i,i)
    var
      max_diff = (B - u.dot(S).dot(vt)).data_buffer.map(x=>x.abs).max
    echo "Test " & $i_test & " max diff about svd (B == U.dot(S).dot(VT)): \nmax_diff=" & $max_diff
    doAssert(max_diff < 1e-10)
    echo " ... good."

  for i_test in 0 .. 10:
    var
      A = np.random.normal(0,1,@[1,50])
      B = A.ravel().tolist().to(seq[float]).toNdArray(A.shape.to(seq[int]))
      (u, s, vt) = B.svd
      S = newSeq[float](B.data_buffer.len).toNdArray(B.shape)
    for i in 0..<min(B.shape):
      S.set_at(s.at(i),i,i)
    var
      max_diff = (B - u.dot(S).dot(vt)).data_buffer.map(x=>x.abs).max
    echo "Test " & $i_test & " max diff about svd (B == U.dot(S).dot(VT)): \nmax_diff=" & $max_diff
    doAssert(max_diff < 1e-10)
    echo " ... good."

  # for i_test in 0 .. 10:
  #   var
  #     A = np.random.normal(0,1,@[30,1])
  #     B = A.ravel().tolist().to(seq[float]).toNdArray(A.shape.to(seq[int]))
  #     # B = normal(@[30,1])
  #     (u, s, vt) = B.svd
  #     S = newSeq[float](B.data_buffer.len).toNdArray(B.shape)
  #   for i in 0..<min(B.shape):
  #     S.set_at(s.at(i),i,i)
  #   # echo B
  #   echo u
  #   echo S
  #   echo vt
  #   var
  #     max_diff = (B - u.dot(S).dot(vt)).data_buffer.map(x=>x.abs).max
  #   echo "Test " & $i_test & " max diff about svd (B == U.dot(S).dot(VT)): \nmax_diff=" & $max_diff
  #   doAssert(max_diff < 1e-5)
  #   echo " ... good."

  ####################
  # Test det
  for i_test in 0 .. 10:
    var
      A = np.random.normal(0,1,@[10,10])
      B = A.ravel().tolist().to(seq[float]).toNdArray(A.shape.to(seq[int]))
      rst1 = np.linalg.det(A).to(float)
      rst2 = B.det
      diff = rst1 - rst2
    echo "Test " & $i_test & " max diff about determinant (compare with numpy): \ndiff = " & $diff
    doAssert(diff.abs < 1e-10)
    echo " ... good."

  for i_test in 0 .. 10:
    var
      A = np.random.normal(0,1,@[1,1])
      B = A.ravel().tolist().to(seq[float]).toNdArray(A.shape.to(seq[int]))
      rst1 = np.linalg.det(A).to(float)
      rst2 = B.det
      diff = rst1 - rst2
    echo "Test " & $i_test & " max diff about determinant (compare with numpy): \ndiff = " & $diff
    doAssert(diff.abs < 1e-10)
    echo " ... good."

  ####################
  # Test qr
  for i_test in 0 .. 10:
    var
      A = np.random.normal(0,1,@[10,10])
      B = A.ravel().tolist().to(seq[float]).toNdArray(A.shape.to(seq[int]))
      rst1 = np.linalg.det(A).to(float)
      rst2 = B.det
      diff = rst1 - rst2
    echo "Test " & $i_test & " max diff about determinant (compare with numpy): \ndiff = " & $diff
    doAssert(diff.abs < 1e-10)
    echo " ... good."

  for i_test in 0 .. 10:
    var
      A = np.random.normal(0,1,@[1,1])
      B = A.ravel().tolist().to(seq[float]).toNdArray(A.shape.to(seq[int]))
      rst1 = np.linalg.det(A).to(float)
      rst2 = B.det
      diff = rst1 - rst2
    echo "Test " & $i_test & " max diff about determinant (compare with numpy): \ndiff = " & $diff
    doAssert(diff.abs < 1e-10)
    echo " ... good."

  ####################
  # Test matrix rank
  for i_test in 0 .. 10:
    var
      A = np.random.normal(0,1,@[10,10])
      B = A.ravel().tolist().to(seq[float]).toNdArray(A.shape.to(seq[int]))
      rst1 = np.linalg.matrix_rank(A).to(int)
      rst2 = B.matrix_rank
      diff = rst1 - rst2
    echo "Test " & $i_test & " max diff about matrix rank (compare with numpy): \ndiff = " & $diff
    doAssert(diff.abs == 0)
    echo " ... good."

  for i_test in 0 .. 10:
    var
      A = np.random.normal(0,1,@[1000,10])
      B = A.ravel().tolist().to(seq[float]).toNdArray(A.shape.to(seq[int]))
      rst1 = np.linalg.matrix_rank(A).to(int)
      rst2 = B.matrix_rank
      diff = rst1 - rst2
    echo "Test " & $i_test & " max diff about matrix rank (compare with numpy): \ndiff = " & $diff
    doAssert(diff.abs == 0)
    echo " ... good."

  for i_test in 0 .. 10:
    var
      A = np.random.normal(0,1,@[2,1000])
      B = A.ravel().tolist().to(seq[float]).toNdArray(A.shape.to(seq[int]))
      rst1 = np.linalg.matrix_rank(A).to(int)
      rst2 = B.matrix_rank
      diff = rst1 - rst2
    echo "Test " & $i_test & " max diff about matrix rank (compare with numpy): \ndiff = " & $diff
    doAssert(diff.abs == 0)
    echo " ... good."


  ####################
  # Test dot1
  for i_test in 0 .. 10:
    var
      A = np.random.normal(0,1,@[20,10])
      B = np.random.normal(0,1,@[10,30])
      AA = A.ravel().tolist().to(seq[float]).toNdArray(A.shape.to(seq[int]))
      BB = B.ravel().tolist().to(seq[float]).toNdArray(B.shape.to(seq[int]))
      rstShape = @[A.shape.to(seq[int])[0], B.shape.to(seq[int])[1]]
      rst1 = A.dot(B).ravel().tolist().to(seq[float]).toNdArray(rstShape)
      rst2 = AA.dot(BB)
      max_diff = (rst1-rst2).data_buffer.map(x=>x.abs).max
    echo "Test " & $i_test & " max diff about dot product 1 (compare with numpy): \nmax_diff = " & $max_diff
    doAssert(max_diff.abs < 1e-10)
    echo " ... good."

  ####################
  # Test dot2
  for i_test in 0 .. 10:
    var
      A = np.random.normal(0,1,@[1,10])
      B = np.random.normal(0,1,@[10,30])
      AA = A.ravel().tolist().to(seq[float]).toNdArray(A.shape.to(seq[int]))
      BB = B.ravel().tolist().to(seq[float]).toNdArray(B.shape.to(seq[int]))
      rstShape = @[A.shape.to(seq[int])[0], B.shape.to(seq[int])[1]]
      rst1 = A.dot(B).ravel().tolist().to(seq[float]).toNdArray(rstShape)
      rst2 = AA.dot(BB)
      max_diff = (rst1-rst2).data_buffer.map(x=>x.abs).max
    echo "Test " & $i_test & " max diff about dot product 2 (compare with numpy): \nmax_diff = " & $max_diff
    doAssert(max_diff.abs < 1e-10)
    echo " ... good."
  
  ####################
  # Test dot3
  for i_test in 0 .. 10:
    var
      A = np.random.normal(0,1,@[30,10])
      B = np.random.normal(0,1,@[10,1])
      AA = A.ravel().tolist().to(seq[float]).toNdArray(A.shape.to(seq[int]))
      BB = B.ravel().tolist().to(seq[float]).toNdArray(B.shape.to(seq[int]))
      rstShape = @[A.shape.to(seq[int])[0], B.shape.to(seq[int])[1]]
      rst1 = A.dot(B).ravel().tolist().to(seq[float]).toNdArray(rstShape)
      rst2 = AA.dot(BB)
      max_diff = (rst1-rst2).data_buffer.map(x=>x.abs).max
    echo "Test " & $i_test & " max diff about dot product 3 (compare with numpy): \nmax_diff = " & $max_diff
    doAssert(max_diff.abs < 1e-10)
    echo " ... good."

  ####################
  # Test eigvals
  for i_test in 0 .. 10:
    var
      A = np.random.normal(0,1,@[100,100])
      B = A.ravel().tolist().to(seq[float]).toNdArray(A.shape.to(seq[int]))
      rst1R = np.linalg.eigvals(A).real.tolist().to(seq[float]).toNdArray
      rst1I = np.linalg.eigvals(A).imag.tolist().to(seq[float]).toNdArray
      rst2R = B.eigvals[0]
      rst2I = B.eigvals[1]
      max_diffR = (rst1R-rst2R).data_buffer.map(x=>x.abs).max
      max_diffI = (rst1I-rst2I).data_buffer.map(x=>x.abs).max
    echo "Test " & $i_test & " max diff about eigvals (compare with numpy): \nmax_diffR = " & $max_diffR & ", max_dffI = " & $max_diffI
    doAssert(max_diffR.abs < 1e-5 and max_diffI.abs < 1e-5)
    echo " ... good."

run()