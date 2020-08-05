import ../common
import ../ndarray/ndarray
import ../ndarray/ndarray_masks
import ../ndarray/accessers
import ../ndarray/ndarray_indexers
import typeinfo

proc astype*[T;U](input: NdArray[T], typeinfo: typedesc[U]): NdArray[U]=
  var
    data_buffer = input.toSeq
  result = data_buffer.astype(U).toNdArray(input.shape)

proc where*[T;U](mask: NdArray[bool], val_true: T, val_false: U): NdArray[T]=
  result = newSeq[T](prodOfSeq(mask.shape)).toNdArray(mask.shape)
  result[mask] = val_true
  result[~mask] = val_false.T

proc where*[T;U](mask: NdArray[bool], val_true: NdArray[T], val_false: U): NdArray[T]=
  assert(mask.shape == val_true.shape)
  result = val_true.copy
  result[~mask] = val_false.T

proc where*[T;U](mask: NdArray[bool], val_true: T, val_false: NdArray[U]): NdArray[T]=
  assert(mask.shape == val_false.shape)
  result = newSeq[T](prodOfSeq(mask.shape)).toNdArray(mask.shape)
  result[mask] = val_true
  if $T == $U:
    result[~mask]= val_false
  else:
    result[~mask]= val_false.astype(T)[~mask]

proc where*[T;U](mask: NdArray[bool], val_true: NdArray[T], val_false: NdArray[U]): NdArray[T]=
  assert(mask.shape == val_true.shape)
  assert(mask.shape == val_false.shape)
  result = val_true.copy
  if $T == $U:
    result[~mask] = val_false
  else:
    result[~mask] = val_false.astype(T)

proc concat*[T](arrs: varargs[NdArray[T]], axis: int): NdArray[T]=
  assert(axis < arrs[0].shape.len)
  var
    base_shape = arrs[0].shape
  base_shape[axis] = 0

  for i in 0 ..< arrs.len:
    var
      another_shape = arrs[i].shape
    another_shape[axis] = 0
    assert(base_shape == another_shape)
  
  var
    shape = arrs[0].shape
  shape[axis] = 0
  for i in 0 ..< arrs.len:
    shape[axis] += arrs[i].shape[axis]
  var
    fake_indexer = get_indexer_list(shape)
  
  result = newSeq[T](prodOfSeq(shape)).toNdArray(shape)
  var
    lowIdx = 0
  for i in 0 ..< arrs.len:
    fake_indexer[axis].lowIdx = lowIdx
    fake_indexer[axis].highIdx = lowIdx + arrs[i].shape[axis] - 1
    result.assign_sliced(arrs[i], fake_indexer)
    lowIdx = fake_indexer[axis].highIdx + 1

when isMainModule:
  var
    A = @[1,2,3,4,5,6].astype(float).toNdArray(@[2,3])
    B = @[5,6,7,8].astype(float).toNdArray(@[2,2])
  echo A.concat(B, axis=1)