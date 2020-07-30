import ndarray
import ndarray_indexers
import sequtils
import strformat
import sets
import ../common

proc get_bool_mask[T](input: NdArray[T], value:T, op: string): NdArray[bool]=
  var
    data_buffer = newSeq[bool](prodOfSeq(input.shape))
    coordindates = get_all_coordinates(get_indexer_list(input.shape))
  if op == ">" :
    for i, coordinate in coordindates:
      data_buffer[i] = input.data_buffer[get_location_from_coordinate(input.start_idx_in_buffer, input.strides, coordinate)] >  value
  elif op == ">=" :
    for i, coordinate in coordindates:
      data_buffer[i] = input.data_buffer[get_location_from_coordinate(input.start_idx_in_buffer, input.strides, coordinate)] >=  value
  elif op == "<":
    for i, coordinate in coordindates:
      data_buffer[i] = input.data_buffer[get_location_from_coordinate(input.start_idx_in_buffer, input.strides, coordinate)] <  value
  elif op == "<=":
    for i, coordinate in coordindates:
      data_buffer[i] = input.data_buffer[get_location_from_coordinate(input.start_idx_in_buffer, input.strides, coordinate)] <=  value
  elif op == "!=":
    for i, coordinate in coordindates:
      data_buffer[i] = input.data_buffer[get_location_from_coordinate(input.start_idx_in_buffer, input.strides, coordinate)] !=  value
  elif op == "==":
    for i, coordinate in coordindates:
      data_buffer[i] = input.data_buffer[get_location_from_coordinate(input.start_idx_in_buffer, input.strides, coordinate)] ==  value
  result = data_buffer.toNdArray(input.shape)

proc get_bool_mask_float[T:SomeFloat](input: NdArray[T], value:T, op: string): NdArray[bool]=
  var
    data_buffer = newSeq[bool](prodOfSeq(input.shape))
    coordindates = get_all_coordinates(get_indexer_list(input.shape))
  if op == "isnan":
    for i, coordinate in coordindates:
      data_buffer[i] = input.data_buffer[get_location_from_coordinate(input.start_idx_in_buffer, input.strides, coordinate)] ==  NaN
  elif op == "isinf":
    for i, coordinate in coordindates:
      data_buffer[i] = input.data_buffer[get_location_from_coordinate(input.start_idx_in_buffer, input.strides, coordinate)] ==  Inf
  elif op == "isfinite":
    for i, coordinate in coordindates:
      data_buffer[i] = input.data_buffer[get_location_from_coordinate(input.start_idx_in_buffer, input.strides, coordinate)] != NaN and input.data_buffer[get_location_from_coordinate(input.start_idx_in_buffer, input.strides, coordinate)] != Inf
  result = data_buffer.toNdArray(input.shape)

proc `>`*[T;U](input: NdArray[T], value:U): NdArray[bool]=
  result = input.get_bool_mask(value.T, ">")

proc `>`*[T;U](value:U, input: NdArray[T]): NdArray[bool]=
  result = input.get_bool_mask(value.T, "<")

proc `>=`*[T;U](input: NdArray[T], value:U): NdArray[bool]=
  result = input.get_bool_mask(value.T, ">=")

proc `>=`*[T;U](value:U, input: NdArray[T]): NdArray[bool]=
  result = input.get_bool_mask(value.T, "<=")

proc `<`*[T;U](input: NdArray[T], value:U): NdArray[bool]=
  result = input.get_bool_mask(value.T, "<")

proc `<`*[T;U](value:U, input: NdArray[T]): NdArray[bool]=
  result = input.get_bool_mask(value.T, ">")

proc `<=`*[T;U](input: NdArray[T], value:U): NdArray[bool]=
  result = input.get_bool_mask(value.T, "<=")

proc `<=`*[T;U](value:U, input: NdArray[T]): NdArray[bool]=
  result = input.get_bool_mask(value.T, ">=")

proc `!=`*[T;U](input: NdArray[T], value:U): NdArray[bool]=
  result = input.get_bool_mask(value.T, "!=")

proc `!=`*[T;U](value:U, input: NdArray[T]): NdArray[bool]=
  result = input.get_bool_mask(value.T, "!=")

proc `==`*[T](A,B: NdArray[T]): NdArray[bool]=
  assert(A.shape == B.shape, fmt"Shape are not the same. How to compare?")
  var
    data_buffer = newSeq[bool](prodOfSeq(A.shape))
    coordinates = get_all_coordinates(get_indexer_list(A.shape))
    locationA, locationB : int

  for i, coordinate in coordinates:
    locationA = get_location_from_coordinate(A.start_idx_in_buffer, A.strides, coordinate)
    locationB = get_location_from_coordinate(B.start_idx_in_buffer, B.strides, coordinate)
    data_buffer[i] = (A.data_buffer[locationA] == B.data_buffer[locationB])

  result = data_buffer.toNdArray(A.shape)

proc `==`*[T;U](input: NdArray[T], value:U): NdArray[bool]=
  result = input.get_bool_mask(value.T, "==")

proc `==`*[T;U](value:U, input: NdArray[T]): NdArray[bool]=
  result = input.get_bool_mask(value.T, "==")

proc isnan*[T: SomeFloat](input: NdArray[T]): NdArray[bool]=
  result = input.get_bool_mask_float(0.T, "isnan")

proc isinf*[T: SomeFloat](input: NdArray[T]): NdArray[bool]=
  result = input.get_bool_mask_float(0.T, "isinf")

proc isfinite*[T: SomeFloat](input: NdArray[T]): NdArray[bool]=
  result = input.get_bool_mask_float(0.T, "isfinite")

proc get_coordinates_from_mask(mask: NdArray[bool]): seq[seq[int]]=
  var
    all_coordinates = get_all_coordinates(get_indexer_list(mask.shape))
    location: int
  result = @[]
  for i, coordinate in all_coordinates:
    location = get_location_from_coordinate(mask.start_idx_in_buffer, mask.strides, coordinate)
    if mask.data_buffer[location]:
      result.add(coordinate)

proc sum*(mask: NdArray[bool]): int=
  var
    coordinates = get_coordinates_from_mask(mask)
  result = coordinates.len

proc `[]`*[T](input: NdArray[T], mask: NdArray[bool]): NdArray[T]=
  var
    coordinates = get_coordinates_from_mask(mask)
    data_buffer = newSeq[T](coordinates.len)
  for i, coordinate in coordinates:
    data_buffer[i] = input.data_buffer[get_location_from_coordinate(input.start_idx_in_buffer, input.strides, coordinate)]
  result = data_buffer.toNdArray

proc `[]=`*[T](input: NdArray[T], mask: NdArray[bool], another_array: NdArray[T]): NdArray[T]=
  var
    coordinates = get_coordinates_from_mask(mask)
    location, location_in_another_array : int
  assert(coordinates.len == another_array.shape[0] and another_array.shape.len == 1, fmt"dimension not matced when assigning values.")
  for i, coordinate in coordinates:
    location = get_location_from_coordinate(input.start_idx_in_buffer, input.strides, coordinate)
    location_in_another_array = get_location_from_coordinate(another_array.start_idx_in_buffer, another_array.strides, @[i])
    input.data_buffer[location] = another_array.data_buffer[location_in_another_array]
  result = input.clone

proc `[]=`*[T;U](input: NdArray[T], mask: NdArray[bool], value: U) =
  var
    coordinates = get_coordinates_from_mask(mask)
    location : int
  for i, coordinate in coordinates:
    location = get_location_from_coordinate(input.start_idx_in_buffer, input.strides, coordinate)
    input.data_buffer[location] = value.T

#TODO change the following proc to bits operations
proc `&`*(A,B: NdArray[bool]): NdArray[bool]=
  assert(A.shape==B.shape,fmt"Dimension not matched.")
  var
    data_buffer = newSeq[bool](prodOfSeq(A.shape))
    coordinatesA = get_coordinates_from_mask(A).toHashSet
    coordinatesB = get_coordinates_from_mask(B).toHashSet
    coordinates = coordinatesA * coordinatesB
    location: int
    strides = get_strides_from_shape(A.shape)
  for coordinate in coordinates:
    location = get_location_from_coordinate(0, strides, coordinate)
    data_buffer[location] = true
  result = data_buffer.toNdArray(A.shape)

proc `|`*(A,B: NdArray[bool]): NdArray[bool]=
  assert(A.shape==B.shape,fmt"Dimension not matched.")
  var
    data_buffer = newSeq[bool](prodOfSeq(A.shape))
    coordinatesA = get_coordinates_from_mask(A).toHashSet
    coordinatesB = get_coordinates_from_mask(B).toHashSet
    coordinates = coordinatesA + coordinatesB
    location: int
    strides = get_strides_from_shape(A.shape)
  for coordinate in coordinates:
    location = get_location_from_coordinate(0, strides, coordinate)
    data_buffer[location] = true
  result = data_buffer.toNdArray(A.shape)

proc `~`*(A: NdArray[bool]): NdArray[bool]=
  var
    coordinates = get_all_coordinates(get_indexer_list(A.shape))
    strides = get_strides_from_shape(A.shape)
    location : int
  result = A.copy

  for coordinate in coordinates:
    location = get_location_from_coordinate(0, strides, coordinate)
    result.data_buffer[location] = not result.data_buffer[location]

when isMainModule:
  var
    a = @[1,2,3,4].toNdArray(@[2,2])
    b = @[0,2,3,4].toNdArray(@[2,2])
  # echo a == b
  echo a
  echo (a > 2)
  echo (~(a > 2)) & (a<2)

  # echo @[1,1,2].toHashSet