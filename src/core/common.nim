import strutils
import strformat

proc astype*[T;U: not string](input : seq[T], dtype : typedesc[U]) : seq[U] =
  result = newSeq[U](input.len)
  for idx in 0..<input.len:
    result[idx] = input[idx].U

proc asString*[T](input : seq[T]) : seq[string] =
  result = newSeq[string](input.len)
  for idx in 0..<input.len:
    result[idx] = `$`(input[idx])

proc astype*[T;U: not string](input : openArray[T], dtype : typedesc[U]) : seq[U] =
  result = newSeq[U](input.len)
  for idx in 0..<input.len:
    result[idx] = input[idx].U

proc asString*[T](input : openArray[T]) : seq[string] =
  result = newSeq[string](input.len)
  for idx in 0..<input.len:
    result[idx] = `$`(input[idx])

proc flattenSeq*[T](input: seq[T]): seq[T] =
  return input

proc flattenSeq*[T](input: seq[seq[T]]): auto =
  return input.concat.flattenSeq

proc getShapeOfSeq*[T](input: seq[T]): seq[int] =
  return @[input.len]

proc getShapeOfSeq*[T](input: seq[seq[T]]): auto =
  if input.len>0:
    result = @[input.len].concat(input[0].getShapeOfSeq)
  else:
    result = @[0]

proc prodOfSeq*[T](input: seq[T]): T=
  result = 1
  for item in input:
    result *= item

proc xrange*[T](start, stop, step: T, inclusive: bool = false): seq[T]=
  assert(step!=0, fmt"step size cannot be 0.")
  result = @[]
  var idx = start

  if step > 0:
    if inclusive:
      while idx <= stop:
        result.add(idx)
        idx += step
    else:
      while idx < stop:
        result.add(idx)
        idx += step
  else:
    if inclusive:
      while idx >= stop:
        result.add(idx)
        idx += step
    else:
      while idx > stop:
        result.add(idx)
        idx += step

proc xrange*[T](stop: T): seq[T]=
  result = xrange(0.T, stop, 1.T)

proc cmp_seq_lexical*[T](list1: seq[T], list2: seq[T]): int=
  # test if list1 < list2
  assert(list1.len == list2.len and list1.low == list2.low and list1.high == list2.high, fmt"to compare two sequence lists, their length must be equal.")
  result = 1
  for idx in list1.low .. list1.high:
    if list1[idx] < list2[idx]:
      return 0

proc get_strides_from_shape*(shape: seq[int]): seq[int]=
  if shape.len == 1:
    result = @[1]
  elif shape.len == 2:
    result = @[shape[1],1]
  else:
    result = newSeq[int](shape.len)
    for idx in countdown(shape.len-1,0):
      if idx == shape.len-1:
        result[idx] = 1
      else:
        result[idx] = result[idx+1] * shape[idx+1]

proc addIndent(input: string, length: int): string=
  var lines = input.splitLines
  for idx, line in lines:
    lines[idx] = " ".repeat(length) & line
  result = lines.join("\n")

proc print1DSeq[T](input: seq[T]): string=
  result = "["
  if input.len <= 10:
    for idx, item in input:
      result &= $item
      if idx < input.len-1: result &= ", "
  else:
    for idx, item in input[0..4]:
      result &= $item & ", "
    result &= "..., "
    for idx, item in input[input.len-5..input.len-1]:
      result &= $item
      if idx < 4: result &= ", "
  result &= "]" 

proc printNdSeq*[T](input: seq[T], shape:seq[int]): string=
  assert(shape.len>0, fmt"to pretty-print a sequence, shape must be positive.")
  assert(prodOfSeq(shape)==input.len, fmt"given shape is not compatible with input sequence.")
  var strides = get_strides_from_shape(shape)

  var axis_0_len, axis_0_stride: int
  var axis_other_len: seq[int]
  var rows: seq[string]

  if shape.len == 1:
    result = print1DSeq[T](input)
  else:
    axis_0_len = shape[0]
    axis_0_stride = strides[0]
    axis_other_len = shape[1..shape.len-1]

    result = "[\n"
    if axis_0_len <= 10:
      rows = newSeq[string](axis_0_len)
      for idx in 0..axis_0_len-1:
        rows[idx] = printNdSeq(input[idx*axis_0_stride..<(idx+1)*axis_0_stride], axis_other_len)
        rows[idx] = rows[idx].addIndent(4)
      result &= rows.join(",\n")
    else:
      rows = newSeq[string](11)
      for idx in 0..4:
        rows[idx] = printNdSeq(input[(idx*axis_0_stride)..<((idx+1)*axis_0_stride)], axis_other_len)
        rows[idx] = rows[idx].addIndent(4)
      rows[5] =  " ".repeat(4) & "..."
      for idx in axis_0_len-5..axis_0_len-1:
        rows[idx-axis_0_len+11] = printNdSeq(input[(idx*axis_0_stride)..<((idx+1)*axis_0_stride)], axis_other_len)
        rows[idx-axis_0_len+11] = rows[idx-axis_0_len+11].addIndent(4)
      result &= rows.join(",\n")
    result &= "\n]"

when isMainModule:
  var
    a = @[1,2,3]
    b = a.asString
    c = a.astype(float)
  echo b
  echo c
  # echo string(1)