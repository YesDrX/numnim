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
  # Test +
  for i_test in 0 .. 10:
    var
      A1 = np.random.normal(0,1,@[1000,100])
      A2 = np.random.normal(0,1,@[1000,100])
      B1 = A1.ravel().tolist().to(seq[float]).toNdArray(A1.shape.to(seq[int]))
      B2 = A2.ravel().tolist().to(seq[float]).toNdArray(A2.shape.to(seq[int]))
      rst1 = np.add(A1,A2).ravel().tolist().to(seq[float]).toNdArray(A1.shape.to(seq[int]))
      rst2 = B1 + B2
      max_diff = (rst1 - rst2).data_buffer.map(x=>x.abs).max
    echo "Test " & $i_test & " max diff about + (compare with numpy): \nmax_diff=" & $max_diff
    doAssert(max_diff < 1e-10)
    echo " ... good."

  # Test -
  for i_test in 0 .. 10:
    var
      A1 = np.random.normal(0,1,@[1000,100])
      A2 = np.random.normal(0,1,@[1000,100])
      B1 = A1.ravel().tolist().to(seq[float]).toNdArray(A1.shape.to(seq[int]))
      B2 = A2.ravel().tolist().to(seq[float]).toNdArray(A2.shape.to(seq[int]))
      rst1 = np.subtract(A1,A2).ravel().tolist().to(seq[float]).toNdArray(A1.shape.to(seq[int]))
      rst2 = B1 - B2
      max_diff = (rst1 - rst2).data_buffer.map(x=>x.abs).max
    echo "Test " & $i_test & " max diff about - (compare with numpy): \nmax_diff=" & $max_diff
    doAssert(max_diff < 1e-10)
    echo " ... good."

  # Test *
  for i_test in 0 .. 10:
    var
      A1 = np.random.normal(0,1,@[1000,100])
      A2 = np.random.normal(0,1,@[1000,100])
      B1 = A1.ravel().tolist().to(seq[float]).toNdArray(A1.shape.to(seq[int]))
      B2 = A2.ravel().tolist().to(seq[float]).toNdArray(A2.shape.to(seq[int]))
      rst1 = np.multiply(A1,A2).ravel().tolist().to(seq[float]).toNdArray(A1.shape.to(seq[int]))
      rst2 = B1 * B2
      max_diff = (rst1 - rst2).data_buffer.map(x=>x.abs).max
    echo "Test " & $i_test & " max diff about * (compare with numpy): \nmax_diff=" & $max_diff
    doAssert(max_diff < 1e-10)
    echo " ... good."

  # Test /
  for i_test in 0 .. 10:
    var
      A1 = np.random.normal(0,1,@[1000,100])
      A2 = np.random.normal(0,1,@[1000,100])
      B1 = A1.ravel().tolist().to(seq[float]).toNdArray(A1.shape.to(seq[int]))
      B2 = A2.ravel().tolist().to(seq[float]).toNdArray(A2.shape.to(seq[int]))
      rst1 = np.divide(A1,A2).ravel().tolist().to(seq[float]).toNdArray(A1.shape.to(seq[int]))
      rst2 = B1 / B2
      max_diff = (rst1 - rst2).data_buffer.map(x=>x.abs).max
    echo "Test " & $i_test & " max diff about / (compare with numpy): \nmax_diff=" & $max_diff
    doAssert(max_diff < 1e-10)
    echo " ... good."


run()