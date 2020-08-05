import sequtils
import math
import ../ndarray/ndarray
import ../ndarray/ndarray_indexers
import ../ndarray/ndarray_operators
import ../ndarray/accessers
import ../common

proc quickSelect*[T](a: var openarray[T]; k: int, inl = 0, inr = -1): T =
  var
    r = if inr >= 0: inr else: a.len - 1
    st = 0
  for i in 0 ..< r:
    if a[i] > a[r]: continue
    swap a[i], a[st]
    inc st
  swap a[r], a[st]
  if k == st: 
    return a[st]
  elif st > k:
    return quickSelect(a, k, 0, st - 1)
  else:
    return quickSelect(a, k, st, inr)

proc makeCopyWihtoutNaNs*[T](a: openArray[T], offset = 0, stride = 1): seq[T]=
  var
    numNaNs = 0
    full_len = 0
  if T is SomeFloat:
    result = newSeq[T]((a.len - offset) div stride)
    for i in 0 ..< ((a.len - offset) div stride):
      if a[i * stride + offset].classify != fcNan:
        result[i-numNaNs] = a[i * stride + offset]
      else:
        numNaNs += 1
  else:
    result = @a
  full_len = result.len
  if numNaNs>0: result.delete((full_len - numNaNs).Natural, (full_len - 1).Natural)  

proc quickSelectWithNaN*[T](a: openArray[T], k: int, offset = 0, stride = 1): T=
  var
    b = makeCopyWihtoutNaNs(a, offset, stride)
  return quickSelect(b, k, 0, -1)

proc nanpercentile*[T;U](a: openArray[T], pct: U, offset = 0, stride = 1, how = "left"): T=
  var
    b = makeCopyWihtoutNaNs(a, offset, stride)
    lb = (b.len.float * pct.float / 100.float).floor.int
    ub = (b.len.float * pct.float / 100.float).ceil.int
  
  if how == "left":
    return quickSelect(b, lb)
  elif how == "right":
    return quickSelect(b, ub)
  elif how == "average":
    var
      c = b
      rst1 = quickSelect(b, lb)
      rst2 = quickSelect(c, ub)
    return (rst1 + rst2) / 2.T

proc nanpercentile*[T;U](a: NdArray[T], pct: U, how = "left"): T=
  result = nanpercentile(a.flatten.data_buffer, pct, 0, 1, how)

proc nanpercentile*[T;U](a: NdArray[T], pcts: seq[U], how = "left"): seq[T]=
  result = @[]
  for pct_item in pcts:
    result.add(nanpercentile(a.data_buffer, pct_item, a.start_idx_in_buffer, a.strides[0], how))

proc nanpercentile*[T;U](a: NdArray[T], pct: U, axis : int, how = "left"): NdArray[T]=
  assert(axis < a.shape.len)
  var
    data_buffer = newSeq[T](a.shape[axis])
    indexer_list = get_indexer_list(a.shape)
    shape = @[a.shape[axis]]
  for i in 0 ..< shape[0]:
    indexer_list[axis].lowIdx = i
    indexer_list[axis].highIdx = i
    data_buffer[i] = (nanpercentile(a.view_sliced(indexer_list).copy.data_buffer, pct, 0, 1, how))
  result  = data_buffer.toNdArray(shape)

when isMainModule:
  import ../ndarray/random_ndarray
  import sorting
  var
    A = normal(@[4,10])
  echo A
  echo A.nanpercentile(50, axis=0)
  echo A[0].sort
  # echo x.nanpercentile(50, axis = 0)
  # echo x