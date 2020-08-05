import sugar
import random
import strformat
import ../ndarray/ndarray
import ../ndarray/ndarray_indexers
import ../common

#########################################
# Sort Sequence
#########################################
proc arrayQuickSorted*[T](a: var openarray[T], inl = 0, inr = -1, offset = 0, stride = 1, cmp: proc(x, y : T): bool) =
  var
    r = if inr >= 0: inr else: ((a.len - offset) div stride) - 1
    l = inl
  let
    n = r - l + 1
    p = l + ((r - l).float * rand(1.0)).int
    pivot = a[p * stride + offset]

  if n < 2: return
  while l <= r:
    if cmp(a[l * stride + offset], pivot):
      inc l
      continue
    if cmp(pivot, a[r * stride + offset]):
      dec r
      continue
    if l <= r:
      swap a[l * stride + offset], a[r * stride + offset]
      inc l
      dec r
  arrayQuickSorted(a, inl, r, offset, stride, cmp)
  arrayQuickSorted(a, l, inr, offset, stride, cmp)

proc quickSorted*[T](a: var openArray[T]) =
  arrayQuickSorted(a, 0, -1, 0, 1, (x,y) => (x<y))

proc merge[T](a, b: var openarray[T], left, middle, right, offset, stride: int, cmp: proc(x, y : T): bool) =
  let
    leftLen = middle - left
    rightLen = right - middle
  var
    l = 0
    r = leftLen
  
  for i in left ..< middle:
    b[l] = a[i * stride + offset]
    inc l
  for i in middle ..< right:
    b[r] = a[i * stride + offset]
    inc r
 
  l = 0
  r = leftLen
  var i = left
 
  while l < leftLen and r < leftLen + rightLen:
    if cmp(b[l], b[r]):
      a[i * stride + offset] = b[l]
      inc l
    else:
      a[i * stride + offset] = b[r]
      inc r
    inc i
 
  while l < leftLen:
    a[i * stride + offset] = b[l]
    inc l
    inc i
  while r < leftLen + rightLen:
    a[i * stride + offset] = b[r]
    inc r
    inc i
 
proc arrayMergeSorted*[T](a, b: var openarray[T], left, right, offset, stride : int, cmp: proc(x, y : T): bool) =
  if right - left <= 1: return

  let middle = (left + right) div 2
  arrayMergeSorted(a, b, left, middle, offset, stride, cmp)
  arrayMergeSorted(a, b, middle, right, offset, stride, cmp)
  merge(a, b, left, middle, right, offset, stride, cmp)
 
proc arrayMergeSorted*[T](a: var openarray[T], offset, stride: int, cmp: proc(x, y : T): bool) =
  var b = newSeq[T](a.len div stride)
  arrayMergeSorted(a, b, 0, b.len, offset, stride, cmp)
 
proc mergeSorted*[T](a: var openArray[T]) =
  arrayMergeSorted(a, 0, 1, (x,y:T) => (x<y))

proc arrayShiftDown*[T](a: var openarray[T]; start, ending, offset, stride: int, cmp: proc(x, y : T): bool) =
  var root = start
  while root * 2 + 1 < ending:
    var child = 2 * root + 1
    if child + 1 < ending and cmp(a[child * stride + offset], a[(child + 1) * stride + offset]):
      inc child
    if cmp(a[root * stride + offset], a[child * stride + offset]):
      swap a[child * stride + offset], a[root * stride + offset]
      root = child
    else:
      return
 
proc arrayHeapSorted*[T](a: var openarray[T], offset, stride: int, cmp: proc(x, y : T): bool) =
  let count = (a.len - offset) div stride
  for start in countdown((count - 2) div 2, 0):
    arrayShiftDown(a, start, count, offset, stride, cmp)
  for ending in countdown(count - 1, 1):
    swap a[ending * stride + offset], a[offset]
    arrayShiftDown(a, 0, ending, offset, stride, cmp)

proc heapSorted*[T](a: var openArray[T]) =
  arrayHeapSorted(a, 0, 1, (x,y) => (x<y))

proc argsort*[T](a: seq[T], offset = 0, stride = 1, how = "mergesort"):seq[int] =
  result = newSeq[int](a.len)
  for i in 0 ..< a.len: result[i] = i
  if how == "mergesort":
    arrayMergeSorted(result, offset, stride, (x,y:int) => (a[x] <= a[y]))
  elif how == "quicksort":
    arrayQuickSorted(result, 0, -1, offset, stride, (x,y:int) => (a[x] < a[y]))
  elif how == "heapsort":
    arrayHeapSorted(result, offset, stride, (x,y:int) => (a[x] < a[y]))
  else:
    quit(how & " is not a valid method for argsort. choose mergesort/quicksort/heapsort.")

#########################################
# Sort NdArray
#########################################
proc argsort*[T](input: NdArray[T]): NdArray[int]=
  assert(input.shape.len == 1)
  result = argsort(input.data_buffer, input.start_idx_in_buffer, input.strides[0]).toNdArray

proc sort*[T](input: NdArray[T]): NdArray[T]=
  result = input.toSeq.toNdArray
  quickSorted(result.data_buffer)

# proc argsort*[T](input: NdArray[T], axis : int): NdArray[int]=
#   assert(axis < input.shape.len)
#   result = newSeq[int](prodOfSeq(input.shape)).toNdArray(input.shape)
#   var
#     (coorindates, block_length) = get_memory_blocks_along_axis(input.shape, axis)
#   for i, coorindate in coorindates:

  

when isMainModule:
  import ../ndarray/accessers
  var
    a = @[5,4,4,3,2,1].toNdArray
  echo a.sort
  echo a[a.argsort.data_buffer]
