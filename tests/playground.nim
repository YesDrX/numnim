import nimpy
import sequtils
import strutils
import ../src/numnim
import times
import strformat
import threadpool

let
  os* = pyImport("os")
  sys* = pyImport("sys")
  np* = pyImport("numpy")
  pd* = pyImport("pandas")
  py* = pyBuiltinsModule()

proc typename*(obj: PyObject): string=
  return  py.type(obj).getAttr("__name__").to(string)

# proc `+`*(obj1, obj2: PyObject): PyObject=
#   return obj1.getAttr("__add__")(obj2)

# var
#   data = np.random.normal(0, 1, @[1000,1000])
#   timeStart = now()

# proc task(data : PyObject) {.thread.} = 
#   let
#     np = pyImport("numpy")
  
#   var
#     tmp = np.linalg.inv(data)

# echo fmt"test starts at {timeStart} ..."
# for i in 0..<4:
#   spawn task(data)
# sync()
# echo fmt"test ends with {now() - timeStart}"

import sugar

var
  npA = np.random.normal(0,1,@[10,10])
  nnA = npA.ravel().tolist().to(seq[float]).toNdArray(@[10,10])
  inv1 = np.linalg.inv(npA)
  inv2 = nnA.inv
  diff = inv1.ravel().tolist().to(seq[float]).toNdArray(@[10,10]) - inv2
echo "Test 1:\n" & fmt"max difference between numnim inv and numpy inv in the random 10by10 matrix is : {diff.data_buffer.map( x => x.abs).max}." 
assert(diff.data_buffer.map( x => x.abs).max < 1e-10)


# type
#   ThreadData = tuple[param: string, param2: int]

# var data = "Hello World" 

# proc showData(data: ThreadData) {.thread.} =
#   echo(data.param, data.param2)

# var thread: Thread[ThreadData]
# createThread[ThreadData](thread, showData, (param: data, param2: 10))
# joinThread(thread)