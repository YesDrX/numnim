import ../ndarray/ndarray
import ../common
import strformat
import strutils

type
  TypedDataFrame*[T] = ref object
    columns* : seq[string]
    rowIndex* : seq[int]
    numRows* : int
    DF*: seq[NdArray[T]]
  
  DataFrame* = ref object
    columns* : seq[string]
    dtypes*: seq[string]
    rowIndex* : seq[int]
    colIdxInTypedDF*: seq[int]
    intDF*: TypedDataFrame[int]
    floatDF*: TypedDataFrame[float]
    boolDF*: TypedDataFrame[bool]
    stringDF*: TypedDataFrame[string]
    numRows: int

proc checkShapeCompatibility[T](df: TypedDataFrame[T], shape: seq[int]): bool=
  result = true
  if df.DF.len>0:
    result = (df.DF[0].shape == shape)
  else:
    quit(fmt"There is no column in the Typed DataFrame.")

proc add[T](tdf: TypedDataFrame[T], columnName: string, columnData: NdArray[T]): TypedDataFrame[T]=
  assert(columnData.shape.len == 1, fmt"We only support 1d array as column for now.")
  if tdf.numRows < 0:
    tdf.numRows = columnData.shape[0]
    tdf.rowIndex = xrange(tdf.numRows)
  else:
    assert(checkShapeCompatibility(tdf, columnData.shape),fmt"Shape of new column is not compatible")
    assert(not tdf.columns.contains(columnName), fmt"{columnName} already exists in the typed dataframe of columns {tdf.columns}.")
  tdf.DF.add(columnData)
  tdf.columns.add(columnName)
  result = tdf

proc `[]`*[T](tdf: TypedDataFrame[T], columnName: string): NdArray[T]=
  assert(tdf.columns.contains(columnName),fmt"{columnName} is not found in the typed dataframe.")
  result = tdf.DF[tdf.columns.find(columnName)]

# proc `[]`*[T](tdf: DataFrame, columnName: string): NdArray[T] =
#   quit(fmt"Nim-lang is static typing. So We cant accessing dynamic typing columns. Instead, if you know the type of {columnName}. Use df.intDF[{columnName}]/df.floatDF[{columnName}]/df.boolDF[{columnName}] instead.")

proc initTypedDataFrame*[T](): TypedDataFrame[T] =
  result = new TypedDataFrame[T]
  result.numRows = -1

proc initDataFrame*(): DataFrame =
  result = new DataFrame
  result.numRows = -1
  result.intDF = initTypedDataFrame[int]()
  result.floatDF = initTypedDataFrame[float]()
  result.boolDF = initTypedDataFrame[bool]()
  result.stringDF = initTypedDataFrame[string]()

proc addColumn*(df: DataFrame, columnName: string, columnData: NdArray[float]): DataFrame=
  df.columns.add(columnName)
  df.dtypes.add(columnData.dtype)
  if df.numRows < 0:
    df.numRows = columnData.shape[0]
    df.colIdxInTypedDF = @[0]
    df.rowIndex = xrange(df.numRows)
  else:
    df.colIdxInTypedDF.add(df.floatDF.DF.len)
  df.floatDF = df.floatDF.add(columnName, columnData)
  assert(df.rowIndex == df.floatDF.rowIndex, fmt"Row index are not matched.")
  result = df

proc addColumn*(df: DataFrame, columnName: string, columnData: NdArray[int]): DataFrame=
  df.columns.add(columnName)
  df.dtypes.add(columnData.dtype)
  if df.numRows < 0:
    df.numRows = columnData.shape[0]
    df.colIdxInTypedDF = @[0]
    df.rowIndex = xrange(df.numRows)
  else:
    df.colIdxInTypedDF.add(df.intDF.DF.len)
  df.intDF = df.intDF.add(columnName, columnData)
  assert(df.rowIndex == df.intDF.rowIndex, fmt"Row index are not matched.")
  result = df

proc addColumn*(df: DataFrame, columnName: string, columnData: NdArray[bool]): DataFrame=
  df.columns.add(columnName)
  df.dtypes.add(columnData.dtype)
  if df.numRows < 0:
    df.numRows = columnData.shape[0]
    df.colIdxInTypedDF = @[0]
    df.rowIndex = xrange(df.numRows)
  else:
    df.colIdxInTypedDF.add(df.boolDF.DF.len)
  df.boolDF = df.boolDF.add(columnName, columnData)
  assert(df.rowIndex == df.boolDF.rowIndex, fmt"Row index are not matched.")
  result = df

proc addColumn*(df: DataFrame, columnName: string, columnData: NdArray[string]): DataFrame=
  df.columns.add(columnName)
  df.dtypes.add(columnData.dtype)
  if df.numRows < 0:
    df.numRows = columnData.shape[0]
    df.colIdxInTypedDF = @[0]
    df.rowIndex = xrange(df.numRows)
  else:
    df.colIdxInTypedDF.add(df.stringDF.DF.len)
  df.stringDF = df.stringDF.add(columnName, columnData)
  assert(df.rowIndex == df.stringDF.rowIndex, fmt"Row index are not matched.")
  result = df

proc addColumn*[T](df: DataFrame, columnName: string, columnData: seq[T]): DataFrame=
  result = df.addColumn(columnName, columnData.toNdArray)

proc addColumn*[T](df: DataFrame, columnName: string, columnData: openArray[T]): DataFrame=
  result = df.addColumn(columnName, columnData.toNdArray)

proc printRowHelper(df: DataFrame, colIdx: int, rowIdx: int): string=
  let
    columnName = df.columns[colIdx]
  if df.dtypes[colIdx] == "string":
    result = df.stringDF[columnName].at(rowIdx)
  elif df.dtypes[colIdx] == "int" or df.dtypes[colIdx] == "int64":
    result = `$`(df.intDF[columnName].at(rowIdx))
  elif df.dtypes[colIdx] == "bool":
    result = `$`(df.boolDF[columnName].at(rowIdx))
  elif df.dtypes[colIdx] == "float" or df.dtypes[colIdx] == "float64":
    result = `$`(df.floatDF[columnName].at(rowIdx))
  else:
    quit("?????")
  if result.len>10:
    result = result[0..9]

proc printRow(df: DataFrame, rowIdx: int): string=
  var
    tmp = ""
  
  result = ""

  if df.dtypes.len > 20:
    for colIdx in 0..<10:
      tmp = df.printRowHelper(colIdx, rowIdx)
      result &= fmt"{tmp:<10}"
    result &= "...       "
    for colIdx in (df.dtypes.len-10)..(df.dtypes.len-1):
      tmp = df.printRowHelper(colIdx, rowIdx)
      result &= fmt"{tmp:<10}"
  else:
    for colIdx in 0..<df.dtypes.len:
      tmp = df.printRowHelper(colIdx, rowIdx)
      result &= fmt"{tmp:<10}"
  result &= "\n"

proc printRow(df: TypedDataFrame, rowIdx: int): string=
  var
    tmp = ""
  
  result = ""

  if df.columns.len > 20:
    for colIdx in 0..<10:
      tmp = `$`(df.DF[colIdx].at(rowIdx))
      result &= fmt"{tmp:<10}"
    result &= "...       "
    for colIdx in (df.columns.len-10)..(df.columns.len-1):
      tmp = `$`(df.DF[colIdx].at(rowIdx))
      result &= fmt"{tmp:<10}"
  else:
    for colIdx in 0..<df.columns.len:
      tmp = `$`(df.DF[colIdx].at(rowIdx))
      result &= fmt"{tmp:<10}"
  result &= "\n"

proc `$`*(df: DataFrame): string=
  var
    printCols = df.columns.len
  if printCols > 20: printCols = 21
  result = "=".repeat(10).repeat(printCols) & "\n"
  if df.columns.len > 20:
    for colIdx in 0..<10: result &= fmt"{df.columns[colIdx]:<10}"
    result &= fmt"...       "
    for colIdx in (df.columns.len-10)..(df.columns.len-1): result &= fmt"{df.columns[colIdx]:<10}"
    result &= "\n"
    for colIdx in 0..<10: result &= fmt"{df.dtypes[colIdx]:<10}"
    result &= fmt"...       "
    for colIdx in (df.columns.len-10)..(df.columns.len-1): result &= fmt"{df.dtypes[colIdx]:<10}"
    result &= "\n"
  else:
    for colIdx in 0..<printCols: result &= fmt"{df.columns[colIdx]:<10}"
    result &= "\n"
    for colIdx in 0..<printCols: result &= fmt"{df.dtypes[colIdx]:<10}"
    result &= "\n"
  result &= "-".repeat(10).repeat(printCols) & "\n"

  if df.numRows >= 10:
    for rowIdx in 0..<5:
      result &= df.printRow(rowIdx)
    result &= "...       ".repeat(printCols) & "\n"
    for rowIdx in (df.numRows-5)..(df.numRows-1):
      result &= df.printRow(rowIdx)
  else:
    for rowIdx in 0..<df.numRows:
      result &= df.printRow(rowIdx)
  result &= "=".repeat(10).repeat(printCols) & "\n"
  result &= fmt"({df.columns.len} columns, {df.numRows} rows)"

proc `$`*[T](df: TypedDataFrame[T]): string=
  var
    printCols = df.columns.len
  if printCols > 20: printCols = 21
  result = '_'.repeat(10).repeat(printCols) & "\n" & fmt"Typed DataFrame : [{$T}]" & "\n"
  result &= "=".repeat(10).repeat(printCols) & "\n"
  if df.columns.len > 20:
    for colIdx in 0..<10: result &= fmt"{df.columns[colIdx]:<10}"
    result &= fmt"...       "
    for colIdx in (df.columns.len-10)..(df.columns.len-1): result &= fmt"{df.columns[colIdx]:<10}"
    result &= "\n"
  else:
    for colIdx in 0..<printCols: result &= fmt"{df.columns[colIdx]:<10}"
    result &= "\n"
  result &= "-".repeat(10).repeat(printCols) & "\n"

  if df.numRows >= 10:
    for rowIdx in 0..<5:
      result &= df.printRow(rowIdx)
    result &= "...       ".repeat(printCols) & "\n"
    for rowIdx in (df.numRows-5)..(df.numRows-1):
      result &= df.printRow(rowIdx)
  else:
    for rowIdx in 0..<df.numRows:
      result &= df.printRow(rowIdx)
  result &= "=".repeat(10).repeat(printCols) & "\n"
  result &= fmt"({df.columns.len} columns, {df.numRows} rows)"

when isMainModule:
  import ../common
  var
    a = arange(10)
    b = arange(10.0)
    c = xrange(10).asString
    d = initDataFrame()
  d = d.addColumn("A",a)
  d = d.addColumn("B",b)
  d = d.addColumn("C",c)

  echo d
  echo `$`(d.intDF)
  # echo d.intDF
  # echo d.floatDF
  # echo d.stringDF
  import sequtils
  echo @["a"].cycle(3)
  # echo d["A"]