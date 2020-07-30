import ndarray
import strutils
import sequtils
import strformat
import algorithm
import ../blas/blas_vector
import ../blas/blas_matrix
from ../blas/blas_wrapper as blas import nil
import ../common
import ndarray_indexers

proc ndarray_memory_block_to_blas_vector*[T](input:NdArray[T], offset: int, stride:int, length: int): BlasVector[T]=
  result = init_BlasVector(T)
  result.data = addr(input.data_buffer[offset])
  shallowCopy(result.data_buffer, input.data_buffer)
  result.stride = stride.cint
  result.offset = offset.cint
  result.size = length.cint
  result.has_nim_seq_data_buffer = true

proc assign_view*[T](base_array, another_array: NdArray[T], indexer_list: seq[Indexer]): NdArray[T]=
  assert(T.sizeof == 4 or T.sizeof == 8, fmt"calling blas function to assign values to ndarray only supports 4/8 bytes length datatypes, because I need force cast them to cfloat/cdouble.")
  var
    left_memory_block_offset: int
    right_memory_block_offset: int
    left_memory_stride: int
    right_memory_stride: int
    left_vector, right_vector: BlasVector[T]
    left_vector_cfloat, right_vector_cfloat: BlasVector[cfloat]
    left_vector_cdouble, right_vector_cdouble: BlasVector[cdouble]

    (sliced_shape, start_coordinate_offset) = get_sliced_shape_and_offset(base_array.start_idx_in_buffer, base_array.strides, indexer_list)
    sliced_view_full_indexer_list = get_indexer_list(sliced_shape)
    (sliced_view_memory_block_coordinates, sliced_view_memory_block_new_coordinates, sliced_view_memory_blocks_axis, sliced_view_memory_blocks_axis_len) = get_memory_blocks(sliced_shape, sliced_view_full_indexer_list)

  result = base_array.slice_view(start_coordinate_offset, sliced_shape)
  assert(sliced_shape == another_array.shape, fmt"shape of assigning matrix ({another_array.shape}) is not compatible with the sliced left side matrix.")
  left_memory_stride = result.strides[sliced_view_memory_blocks_axis]
  right_memory_stride = another_array.strides[sliced_view_memory_blocks_axis]
  for i, coordinate in sliced_view_memory_block_new_coordinates:
    left_memory_block_offset = get_location_from_coordinate(result.start_idx_in_buffer, result.strides, coordinate)
    right_memory_block_offset = get_location_from_coordinate(another_array.start_idx_in_buffer, another_array.strides, coordinate)
    left_vector = ndarray_memory_block_to_blas_vector(result, left_memory_block_offset, left_memory_stride, sliced_view_memory_blocks_axis_len)
    right_vector = ndarray_memory_block_to_blas_vector(another_array, right_memory_block_offset, right_memory_stride, sliced_view_memory_blocks_axis_len)
    
    # copy
    if T.sizeof == 4:
      left_vector_cfloat = left_vector.force_cast(cfloat)
      right_vector_cfloat = right_vector.force_cast(cfloat)
      blas.copy(right_vector_cfloat, left_vector_cfloat)
    elif T.sizeof == 8:
      left_vector_cdouble = left_vector.force_cast(cdouble)
      right_vector_cdouble = right_vector.force_cast(cdouble)
      blas.copy(right_vector_cdouble, left_vector_cdouble)

proc assign_non_view*[T](base_array, another_array: NdArray[T], indexer_list: seq[Indexer]): NdArray[T]=
  # when we cannot break ndarray into memory blocks, we assign values one by one
  var
    coordindates = get_all_coordinates(indexer_list)
    location_in_base_array_data_buffer: int
    location_in_another_array_in_data_buffer: int

  assert(@[coordindates.len] == another_array.shape, fmt"left side ndarray shape ({another_array.shape}) is not compatible with the sliced ndarray to assign to (sliced shape = [{coordindates.len}]).")
  for i, coordinate in coordindates:
    location_in_base_array_data_buffer = base_array.getIdx(coordinate)
    location_in_another_array_in_data_buffer = another_array.getIdx(i)
    base_array.data_buffer[location_in_base_array_data_buffer] = another_array.data_buffer[location_in_another_array_in_data_buffer]

  result = base_array.clone

proc assign_sliced*[T](base_array, another_array: NdArray[T], indexer_list: seq[Indexer]): NdArray[T]=
  assert(not base_array.flags.isReadOnly, fmt"Cannot assign values to read-only ndarray.")
  if base_array.flags.isCopiedFromView:
    {.warning : fmt"You are assigning values to a copy of sliced view.".}
  if indexer_list.check_no_array_indexer:
    result = assign_view[T](base_array, another_array, indexer_list)
  else:
    result = assign_non_view[T](base_array, another_array, indexer_list)

proc slice_view*[T](input: NdArray[T], start_coordinate_offset:int, sliced_shape: seq[int]): NdArray[T]=
  result = input.clone
  result.shape = sliced_shape
  result.start_idx_in_buffer = start_coordinate_offset
  result.data_ptr = addr(input.data_buffer[start_coordinate_offset])

proc slice_non_view*[T](input: NdArray[T], indexer_list: seq[Indexer]): NdArray[T]=
  # when we cannot break ndarray into memory blocks, we create a copy.
  var
    coordindates = get_all_coordinates(indexer_list)
    location_in_base_array_data_buffer: int
    data_buffer = newSeq[T](coordindates.len)
  for i, coordinate in coordindates:
    location_in_base_array_data_buffer = input.getIdx(coordinate)
    data_buffer[i] = input.data_buffer[location_in_base_array_data_buffer]
  result = data_buffer.toNdArray

proc view_sliced*[T](input: NdArray[T], indexer_list: seq[Indexer]): NdArray[T]=
  if indexer_list.check_no_array_indexer:
    var
      (sliced_shape, start_coordinate_offset) = get_sliced_shape_and_offset(input.start_idx_in_buffer, input.strides, indexer_list)
    result = slice_view[T](input, start_coordinate_offset, sliced_shape)
    result.flags.isReadOnly = true
    result.flags.isCopiedFromView = true
  else:
    result = slice_non_view[T](input, indexer_list)
    result.flags.isCopiedFromView = true

proc `[]`*[T;U0](input: NdArray[T], idx0: U0): NdArray[T]=
  var
    indexer_list = get_indexer_list(input.shape, idx0)
  result = input.view_sliced(indexer_list)

proc `[]`*[T;U0;U1](input: NdArray[T], idx0: U0, idx1: U1): NdArray[T]=
  var
    indexer_list = get_indexer_list(input.shape, idx0, idx1)
  result = input.view_sliced(indexer_list)

proc `[]`*[T;U0;U1;U2](input: NdArray[T], idx0: U0, idx1: U1, idx2: U2): NdArray[T]=
  var
    indexer_list = get_indexer_list(input.shape, idx0, idx1, idx2)
  result = input.view_sliced(indexer_list)

proc `[]`*[T;U0;U1;U2;UX](input: NdArray[T], idx0: U0, idx1: U1, idx2: U2, idx3: UX, idxX: varargs[UX]): NdArray[T]=
  var
    indexer_list = get_indexer_list(input.shape, idx0, idx1, idx2, idx3, idxX)
  result = input.view_sliced(indexer_list)

proc `[]=`*[T;U0](input: NdArray[T], idx0: U0, another_array: NdArray[T]): void=
  var
    indexer_list = get_indexer_list(input.shape, idx0)
  discard assign_sliced(input, another_array, indexer_list)

proc `[]=`*[T;U0;U1](input: NdArray[T], idx0: U0, idx1: U1, another_array: NdArray[T]): void=
  var
    indexer_list = get_indexer_list(input.shape, idx0, idx1)
  discard assign_sliced(input, another_array, indexer_list)

proc `[]=`*[T;U0;U1;U2](input: NdArray[T], idx0: U0, idx1: U1, idx2: U2, another_array: NdArray[T]): void=
  var
    indexer_list = get_indexer_list(input.shape, idx0, idx1, idx2)
  discard assign_sliced(input, another_array, indexer_list)

proc `[]=`*[T;U0;U1;U2;UX](input: NdArray[T], idx0: U0, idx1: U1, idx2: U2, idx3: UX, idxX: varargs[UX], another_array: NdArray[T]): void=
  var
    indexer_list = get_indexer_list(input.shape, @[idx0, idx1, idx2, idx3] & idxX)
  discard assign_sliced(input, another_array, indexer_list)

when isMainModule:
  var
    a = xrange(0.0.cdouble,120.0.cdouble,1.0.cdouble).toNdArray(@[12,10])
  a[0..1,0..1] = xrange(-1.0,-4.0,-1.0,inclusive=true).toNdArray(@[2,2])

  echo a
  echo a[0 .. 2,0 .. -1]

  var
    b = a[0 .. 2,0 .. -1]
    c = b[_, 0..1]
  echo c