import strformat
import algorithm
import ../common

type
  PLACEHOLDER = distinct byte

  INDEX_TYPE* = enum
    INT
    SLICE
    ARRAY
    ALL
    NULL

  Indexer* = ref object
    lowIdx: int
    highIdx: int
    idx_seq: seq[int]
    idx_type: INDEX_TYPE

const _* = PLACEHOLDER(0)

proc `..`*(a: PLACEHOLDER, b: int): HSlice[int, int]=
  result.a = 0
  result.b = b

proc `..<`*(a: PLACEHOLDER, b: int): HSlice[int, int]=
  result.a = 0
  result.b = b-1

proc `..`*(a: int, b: PLACEHOLDER): HSlice[int, int]=
  result.a = a
  result.b = -1

proc `$`*(idx: HSlice[int, int]): string=
  result = "HSlice: " & $idx.a & " to " & $idx.b

proc `$`*(idx: Indexer): string=
  result = "NdArray axis indexer: Type = " & $idx.idx_type & " , low = " & $idx.lowIdx & " , high = " & $idx.highIdx

proc init_indexer(): Indexer=
  result = new Indexer
  result.lowIdx = -1
  result.highIdx = -1
  result.idx_seq = @[]
  result.idx_type = INDEX_TYPE.NULL

proc get_indexer(idx: int): Indexer=
  result = new Indexer
  result.lowIdx = idx
  result.highIdx = idx
  result.idx_type = INDEX_TYPE.INT

proc get_indexer(idx: HSLICE[int,int]): Indexer=
  result = new Indexer
  result.lowIdx = idx.a
  result.highIdx = idx.b
  result.idx_type = INDEX_TYPE.SLICE

proc get_indexer(idx: HSLICE[int,BackwardsIndex]): Indexer=
  result = new Indexer
  result.lowIdx = idx.a
  result.highIdx = -idx.b.int
  result.idx_type = INDEX_TYPE.SLICE

proc get_indexer(idx: HSLICE[BackwardsIndex,int]): Indexer=
  result = new Indexer
  result.lowIdx = -idx.a.int
  result.highIdx = idx.b
  result.idx_type = INDEX_TYPE.SLICE

proc get_indexer(idx: HSLICE[BackwardsIndex,BackwardsIndex]): Indexer=
  result = new Indexer
  result.lowIdx = -idx.a.int
  result.highIdx = -idx.b.int
  result.idx_type = INDEX_TYPE.SLICE

proc get_indexer(idx: PLACEHOLDER): Indexer=
  result = new Indexer
  result.lowIdx = 0
  result.highIdx = -1
  result.idx_type = INDEX_TYPE.ALL

proc get_indexer(idx: openArray[int]): Indexer=
  result = new Indexer
  result.idx_seq = @idx
  result.idx_type = INDEX_TYPE.ARRAY

proc get_indexer(idx: seq[int]): Indexer=
  result = new Indexer
  result.idx_seq = idx
  result.idx_type = INDEX_TYPE.ARRAY

proc adjust_slice_indexer_of_negative_values(idx:Indexer, full_length:int): Indexer=
  result = deepCopy(idx)
  if idx.idx_type != INDEX_TYPE.ARRAY:
    if idx.lowIdx < 0: result.lowIdx += full_length
    if idx.highIdx < 0: result.highIdx += full_length
    assert(result.lowIdx >= 0 and result.highIdx >= 0 and result.lowIdx <= full_length and result.highIdx <= full_length, fmt"{result.lowIdx} .. {result.highIdx} is not a valid slicer for sequence with length {full_length}.")

proc get_indexer_list*(shape: seq[int]): seq[Indexer]=
  var
    fake_indexer = get_indexer(0 .. -1)

  result = @[]
  for axis in 0 .. shape.len-1:
    result.add(fake_indexer.adjust_slice_indexer_of_negative_values(shape[axis]))

proc get_indexer_list*[U0](shape: seq[int], idx0: U0): seq[Indexer]=
  var
    fake_indexer = get_indexer(0 .. -1)

  result = @[idx0.get_indexer.adjust_slice_indexer_of_negative_values(shape[0])]
  for axis in 1 .. shape.len-1:
    result.add(fake_indexer.adjust_slice_indexer_of_negative_values(shape[axis]))

proc get_indexer_list*[U0;U1](shape: seq[int], idx0: U0, idx1: U1): seq[Indexer]=
  var
    fake_indexer = get_indexer(0 .. -1)

  result = @[idx0.get_indexer.adjust_slice_indexer_of_negative_values(shape[0]),
             idx1.get_indexer.adjust_slice_indexer_of_negative_values(shape[1])]
  for axis in 2 .. shape.len-1:
    result.add(fake_indexer.adjust_slice_indexer_of_negative_values(shape[axis]))

proc get_indexer_list*[U0;U1;U2](shape: seq[int], idx0: U0, idx1: U1, idx2: U2): seq[Indexer]=
  var
    fake_indexer = get_indexer(0 .. -1)

  result = @[idx0.get_indexer.adjust_slice_indexer_of_negative_values(shape[0]),
             idx1.get_indexer.adjust_slice_indexer_of_negative_values(shape[1]),
             idx2.get_indexer.adjust_slice_indexer_of_negative_values(shape[2])]
  for axis in 3 .. shape.len-1:
    result.add(fake_indexer.adjust_slice_indexer_of_negative_values(shape[axis]))

proc get_indexer_list*[U0;U1;U2;UX](shape: seq[int], idx0: U0, idx1: U1, idx2: U2, idx3: UX, idxX: varargs[UX]): seq[Indexer]=
  var
    fake_indexer = get_indexer(0 .. -1)

  result = @[idx0.get_indexer.adjust_slice_indexer_of_negative_values(shape[0]),
             idx1.get_indexer.adjust_slice_indexer_of_negative_values(shape[1]),
             idx2.get_indexer.adjust_slice_indexer_of_negative_values(shape[2]),
             idx3.get_indexer.adjust_slice_indexer_of_negative_values(shape[3])]

  for i in 0 .. (idxX.len-1):
    result.add(idxX[i].get_indexer.adjust_slice_indexer_of_negative_values(shape[i+4]))

  for axis in (idxX.len+4) .. shape.len-1:
    result.add(fake_indexer.adjust_slice_indexer_of_negative_values(shape[axis]))

proc check_no_array_indexer*(list_indexer: seq[Indexer]): bool=
  result = true
  for axis in 0..<list_indexer.len:
    if list_indexer[axis].idx_type == INDEX_TYPE.ARRAY:
      return false
  
proc check_all_int_indexer*(list_indexer: seq[Indexer]): bool=
  result = true
  for axis in 0..<list_indexer.len:
    if list_indexer[axis].lowIdx != list_indexer[axis].highIdx or list_indexer[axis].idx_type == INDEX_TYPE.ARRAY:
      return false

proc get_location_from_coordinate*(offset: int, strides: seq[int], coordinate: seq[int]): int=
  assert(strides.len == coordinate.len, fmt"invalid coordinate for strides.")
  result = offset
  for axis in 0..<strides.len:
    result += strides[axis] * coordinate[axis]

proc get_all_coordinates*(indexer_list: seq[Indexer]): seq[seq[int]]=
  result = @[]
  if check_all_int_indexer(indexer_list):
    var
      tmp :seq[int] = @[]
    for axis in 0 ..< indexer_list.len:
      tmp.add(@[indexer_list[axis].lowIdx])
    result.add(tmp)
  else:
    for axis in 0 ..< indexer_list.len:
      result.add(xrange(indexer_list[axis].lowIdx, indexer_list[axis].highIdx, 1, inclusive=true))
    result = product(result)
    result.sort(cmp_seq_lexical)

proc get_memory_blocks*(shape:seq[int], indexer_list: seq[Indexer]): (seq[seq[int]], seq[seq[int]], int, int)=
# return : tuple of
#  starting coordinate of each memory block in original ndarray
#  starting coordinate of each memory block in sliced ndarray
#  max length axis
#  max length of the sliced axis

  assert(indexer_list.len == shape.len, fmt"input indexer is not valid for shape = {shape}.")
  
  var
    maxLenAxis = -1
    maxLen = 0
    maxLneAxisStartIdx = -1
    fake_list_indexer = indexer_list
    idx_seq_of_axises : seq[seq[int]] = @[]
    coordindates : seq[seq[int]]
    coordindates_in_sliced_view : seq[seq[int]]

  for axis in 0 ..< shape.len:
    assert((indexer_list[axis].highIdx - indexer_list[axis].lowIdx + 1) >= 1, fmt"empty slice on axis = {axis}.")
    assert(indexer_list[axis].idx_type != INDEX_TYPE.ARRAY, fmt"we cannot get continuous memory blocks with Array slicer.")
    if (indexer_list[axis].highIdx - indexer_list[axis].lowIdx + 1) > maxLen :
      maxLen = indexer_list[axis].highIdx - indexer_list[axis].lowIdx + 1
      maxLenAxis = axis
      maxLneAxisStartIdx = indexer_list[axis].lowIdx

  fake_list_indexer[maxLenAxis].highIdx = fake_list_indexer[maxLenAxis].lowIdx
  coordindates = fake_list_indexer.get_all_coordinates()
  for axis in 0 ..< shape.len:
    fake_list_indexer[axis].highIdx = fake_list_indexer[axis].highIdx - fake_list_indexer[axis].lowIdx
    fake_list_indexer[axis].lowIdx = 0
  coordindates_in_sliced_view = fake_list_indexer.get_all_coordinates()

  result = (coordindates, coordindates_in_sliced_view, maxLenAxis, maxLen)

proc get_sliced_shape_and_offset*(offset:int, strides:seq[int], indexer_list: seq[Indexer]): (seq[int], int)=
  var
    sliced_shape = newSeq[int](indexer_list.len)
    start_coordinate_offset = offset

  for axis in 0..<indexer_list.len:
    sliced_shape[axis] = indexer_list[axis].highIdx - indexer_list[axis].lowIdx + 1
    start_coordinate_offset += indexer_list[axis].lowIdx * strides[axis]

  result = (sliced_shape, start_coordinate_offset)

when isMainModule:
  import strutils

  # var
  #   a = get_indexer_list(@[1,10])
  # echo a
  # echo get_memory_blocks(@[1,10], a)
  # echo @[@[1],@[2,3]].product
  # echo a.get_all_coordinates.len