import ../common
import ../ndarray/ndarray
import ../ndarray/ndarray_masks
import ../ndarray/accessers
import ../ndarray/ndarray_operators
import typeinfo
import basic_ufunc
import sugar
import math

proc mapToDataBuffer*[T;U](data: var seq[T], function: proc (x: T): U) =
  for i in 0 ..< data.len:
    data[i] = function(data[i]).T

proc log*[T](input: NdArray[T]): NdArray[float]=
  result = input.toSeq.astype(float).toNdArray(input.shape)
  mapToDataBuffer(result.data_buffer, (x) => ln(x.float))

proc log1p*[T](input: NdArray[T]): NdArray[float]=
  result = input.toSeq.astype(float).toNdArray(input.shape)
  mapToDataBuffer(result.data_buffer, (x) => ln(x.float+1.0))

proc exp*[T](input: NdArray[T]): NdArray[float]=
  result = input.toSeq.astype(float).toNdArray(input.shape)
  mapToDataBuffer(result.data_buffer, (x) => exp(x.float))

proc expM1*[T](input: NdArray[T]): NdArray[float]=
  result = input.exp - 1

proc sin*[T](input: NdArray[T]): NdArray[float]=
  result = input.toSeq.astype(float).toNdArray(input.shape)
  mapToDataBuffer(result.data_buffer, (x) => sin(x.float))

proc cos*[T](input: NdArray[T]): NdArray[float]=
  result = input.toSeq.astype(float).toNdArray(input.shape)
  mapToDataBuffer(result.data_buffer, (x) => cos(x.float))

proc tan*[T](input: NdArray[T]): NdArray[float]=
  result = input.toSeq.astype(float).toNdArray(input.shape)
  mapToDataBuffer(result.data_buffer, (x) => tan(x.float))

proc tanh*[T](input: NdArray[T]): NdArray[float]=
  result = input.toSeq.astype(float).toNdArray(input.shape)
  mapToDataBuffer(result.data_buffer, (x) => tanh(x.float))

proc abs*[T](input: NdArray[T]): NdArray[float]=
  result = input.toSeq.astype(float).toNdArray(input.shape)
  mapToDataBuffer(result.data_buffer, (x) => abs(x.float))

proc power*[T;U](input: NdArray[T], a: U): NdArray[float]=
  result = input.toSeq.astype(float).toNdArray(input.shape)
  mapToDataBuffer(result.data_buffer, (x) => pow(x.float, a.float))

proc floor*[T](input: NdArray[T]): NdArray[float]=
  result = input.toSeq.astype(float).toNdArray(input.shape)
  mapToDataBuffer(result.data_buffer, (x) => floor(x.float))

proc ceil*[T](input: NdArray[T]): NdArray[float]=
  result = input.toSeq.astype(float).toNdArray(input.shape)
  mapToDataBuffer(result.data_buffer, (x) => ceil(x.float))

proc round*[T](input: NdArray[T]): NdArray[float]=
  result = input.toSeq.astype(float).toNdArray(input.shape)
  mapToDataBuffer(result.data_buffer, (x) => round(x.float))

proc round*[T](input: NdArray[T], places: int): NdArray[float]=
  result = input.toSeq.astype(float).toNdArray(input.shape)
  mapToDataBuffer(result.data_buffer, (x) => round(x.float, places))

proc sign*[T](input: T): float= 
  if input>0:
    result = 1.0
  elif input<0:
    result = -1.0

proc sign*[T](input: NdArray[T]): NdArray[float]=
  result = input.toSeq.astype(float).toNdArray(input.shape)
  mapToDataBuffer(result.data_buffer, (x) => x.sign)

when isMainModule:
  var
    A = @[-1.0,2.0,3.0,4.5666].toNdArray(@[2,2])
  echo A.round(2)
  echo A.sign

  # echo pow(3.float,2.float)
  # echo A.exp
  # echo A.expM1
  # echo A.tanh
