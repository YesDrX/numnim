import threadpool
import cpuinfo
import sugar
import times
import strformat
import ../common
import os

# let MAX_WORKERS = countProcessors() - 1
let MAX_WORKERS = countProcessors()

template runThreadParrallelTask[T](function: untyped, data: seq[T]) =
  # var
  #   blocks = data.len div MAX_WORKERS
  #   startIdx, endIdx : int
  # if blocks * MAX_WORKERS < data.len: blocks += 1
  var
    startTime = now()
  # for blockIdx in 0..<blocks:
  #   # echo "working on block " & $(blockIdx+1) &  "/" & $blocks & " ..."
  #   startIdx = blockIdx * MAX_WORKERS
  #   endIdx = startIdx + MAX_WORKERS - 1
  #   endIdx = min(endIdx, data.len-1)
  #   if startIdx > endIdx: continue
  for i in 0..data.len-1:
    discard spawn function(data[i])
  sync()
  echo "Time used = " & $(now() - startTime)

when isMainModule:
  proc worker(input: int): int=
    sleep(5)
    result = 0

  

  var
    data = xrange(10000)
  runThreadParrallelTask(worker, data)
  
  var
    startTime = now()
  for i in 0..data.len-1:
    discard worker(data[i])
  echo "Time used = " & $(now() - startTime)
