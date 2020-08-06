import ../ndarray/ndarray
import ../common
import strformat
import strutils
import sequtils

type
  TypedDataFrame*[T] = ref object
    columns* : seq[string]
    rowIndex* : seq[int]
    numRows* : int
    indexDF*: DataFrame
    DF*: seq[NdArray[T]]

  DataFrame* = ref object
    columns* : seq[string]
    dtypes*: seq[string]
    rowIndex* : seq[int]
    colIdxInTypedDF*: seq[int]
    indexDF*: DataFrame
    intDF*: TypedDataFrame[int]
    floatDF*: TypedDataFrame[float]
    boolDF*: TypedDataFrame[bool]
    stringDF*: TypedDataFrame[string]
    numRows*: int

proc checkShapeCompatibility*[T](df: TypedDataFrame[T], shape: seq[int]): bool=
  result = true
  if df.DF.len>0:
    result = (df.DF[0].shape == shape)
  elif df.rowIndex.len>0:
    result = df.rowIndex.len == shape[0]
  else:
    quit(fmt"There is no column in the Typed DataFrame.")

proc getColIdx*(df:DataFrame, colName: string): int=
  result = df.columns.find(colName)

proc getColIdx*(df:TypedDataFrame, colName: string): int=
  result = df.columns.find(colName)

proc getColDtype*(df:DataFrame, colName: string): string=
  let
    colIdx = df.getColIdx(colName)
  if colIdx >= 0:
    result = df.dtypes[colIdx]
  else:
    quit(fmt"{colName} is not found in DataFrame.")

proc getColIdxInTypedDF*(df:DataFrame, colName: string): int=
  let
    colIdx = df.getColIdx(colName)
  if colIdx >= 0:
    result = df.colIdxInTypedDF[colIdx]
  else:
    quit(fmt"{colName} is not found in DataFrame.")  

proc initTypedDataFrame*[T](): TypedDataFrame[T] =
  result = new TypedDataFrame[T]
  result.columns = @[]
  result.rowIndex = @[]
  result.numRows = 0
  result.indexDF = nil

proc initDataFrame*(): DataFrame =
  result = new DataFrame
  result.columns = @[]
  result.dtypes = @[]
  result.rowIndex = @[]
  result.numRows = 0
  result.indexDF = nil
  result.intDF = initTypedDataFrame[int]()
  result.floatDF = initTypedDataFrame[float]()
  result.boolDF = initTypedDataFrame[bool]()
  result.stringDF = initTypedDataFrame[string]()

proc `[]`*[T](tdf: TypedDataFrame[T], columnName: string): NdArray[T]=
  assert(tdf.columns.contains(columnName),fmt"{columnName} is not found in the typed dataframe.")
  result = tdf.DF[tdf.columns.find(columnName)]

proc `[]`*[T](tdf: TypedDataFrame[T], columnNames: seq[string]): TypedDataFrame[T]=
  result = initTypedDataFrame[T]()
  for colName in columnNames:
    assert(colName in tdf.columns, fmt"{colName} is not found.")
    discard result.addColumn(colName, tdf[colName])

proc syncIndex*[T](df: DataFrame, tdf: TypedDataFrame[T]) =
  # if df.rowIndex.len == 0 and df.indexDF.isNil and tdf.indexDF.isNil and tdf.rowIndex.len > 0:
  #   shallowCopy(df.rowIndex, tdf.rowIndex)
  #   df.numRows = tdf.numRows
  # elif df.rowIndex.len == 0 and df.indexDF.isNil and not tdf.indexDF.isNil and tdf.rowIndex.len > 0:
  #   assert(tdf.rowIndex.len == tdf.indexDF.numRows)
  #   shallowCopy(df.rowIndex, tdf.rowIndex)
  #   shallowCopy(df.indexDF, tdf.indexDF)
  #   df.numRows = tdf.numRows
  # elif df.rowIndex.len > 0 and df.indexDF.isNil and tdf.indexDF.isNil and tdf.rowIndex.len == 0:
  #   shallowCopy(tdf.rowIndex, df.rowIndex)
  #   tdf.numRows = df.numRows
  # elif df.rowIndex.len > 0 and not df.indexDF.isNil and tdf.indexDF.isNil and tdf.rowIndex.len == 0:
  #   assert(df.rowIndex.len == df.indexDF.numRows)
  #   shallowCopy(tdf.rowIndex, df.rowIndex)
  #   shallowCopy(tdf.indexDF, df.indexDF)
  #   tdf.numRows = df.numRows
  # elif df.rowIndex.len > 0 and tdf.rowIndex.len > 0:
  #   assert(df.rowIndex == tdf.rowIndex)
  #   assert(df.numRows == tdf.numRows)
  #   if df.indexDF.isNil and tdf.indexDF.isNil:
  #     discard
  #   elif df.indexDF.isNil and not tdf.indexDF.isNil:
  #     assert(tdf.rowIndex.len == tdf.indexDF.numRows)
  #     shallowCopy(df.indexDF, tdf.indexDF)
  #   elif not df.indexDF.isNil and tdf.indexDF.isNil:
  #     assert(df.rowIndex.len == df.indexDF.numRows)
  #     shallowCopy(tdf.indexDF, df.indexDF)
  #   else:
  #     assert(df.indexDF == tdf.indexDF)
  # elif (df.rowIndex.len == 0 and not df.indexDF.isNil) or (tdf.rowIndex.len == 0 and not tdf.indexDF.isNil):
  #   quit(fmt"DataFrame index is corrupted!")
  # else:
  #   quit(fmt"You should not be here.") 

  if df.rowIndex.len > 0 and df.indexDF.isNil:
    shallowCopy(tdf.rowIndex, df.rowIndex)
    tdf.numRows = df.numRows
  elif df.rowIndex.len > 0 and not df.indexDF.isNil:
    assert(df.rowIndex.len == df.indexDF.numRows)
    shallowCopy(tdf.rowIndex, df.rowIndex)
    shallowCopy(tdf.indexDF, df.indexDF)
    tdf.numRows = df.numRows

proc syncIndex*(df: DataFrame) =
  syncIndex(df, df.intDF)
  syncIndex(df, df.floatDF)
  syncIndex(df, df.boolDF)
  syncIndex(df, df.stringDF)

proc removeTypeAlias*(typename: string): string=
  if typename == "int64":
    return "int"
  elif typename == "float64":
    return "float"
  else:
    return typename

proc addColumn*[T](tdf: TypedDataFrame[T], columnName: string, columnData: NdArray[T]): TypedDataFrame[T]=
  assert(columnData.shape.len == 1, fmt"We only support 1d array as column for now.")
  if tdf.numRows <= 0:
    tdf.numRows = columnData.shape[0]
    tdf.rowIndex = xrange(tdf.numRows)
  else:
    assert(checkShapeCompatibility(tdf, columnData.shape),fmt"Shape of new column is not compatible")
    assert(not tdf.columns.contains(columnName), fmt"{columnName} already exists in the typed dataframe of columns {tdf.columns}.")
  tdf.DF.add(columnData)
  tdf.columns.add(columnName)
  result = tdf

proc addColumn*(df: DataFrame, columnName: string, columnData: NdArray[float]): DataFrame=
  assert(columnData.shape.len == 1)
  df.columns.add(columnName)
  df.dtypes.add(columnData.dtype.removeTypeAlias)
  if df.numRows <= 0:
    df.numRows = columnData.shape[0]
    df.colIdxInTypedDF = @[0]
    df.rowIndex = xrange(df.numRows)
  else:
    assert(df.numRows == columnData.shape[0])
    df.colIdxInTypedDF.add(df.floatDF.DF.len)
  df.floatDF = df.floatDF.addColumn(columnName, columnData)
  df.syncIndex
  result = df

proc addColumn*(df: DataFrame, columnName: string, columnData: NdArray[int]): DataFrame=
  df.columns.add(columnName)
  df.dtypes.add(columnData.dtype.removeTypeAlias)
  if df.numRows <= 0:
    df.numRows = columnData.shape[0]
    df.colIdxInTypedDF = @[0]
    df.rowIndex = xrange(df.numRows)
  else:
    df.colIdxInTypedDF.add(df.intDF.DF.len)
  df.intDF = df.intDF.addColumn(columnName, columnData)
  df.syncIndex
  result = df

proc addColumn*(df: DataFrame, columnName: string, columnData: NdArray[bool]): DataFrame=
  df.columns.add(columnName)
  df.dtypes.add(columnData.dtype)
  if df.numRows <= 0:
    df.numRows = columnData.shape[0]
    df.colIdxInTypedDF = @[0]
    df.rowIndex = xrange(df.numRows)
  else:
    df.colIdxInTypedDF.add(df.boolDF.DF.len)
  df.boolDF = df.boolDF.addColumn(columnName, columnData)
  df.syncIndex
  result = df

proc addColumn*(df: DataFrame, columnName: string, columnData: NdArray[string]): DataFrame=
  df.columns.add(columnName)
  df.dtypes.add(columnData.dtype)
  if df.numRows <= 0:
    df.numRows = columnData.shape[0]
    df.colIdxInTypedDF = @[0]
    df.rowIndex = xrange(df.numRows)
  else:
    df.colIdxInTypedDF.add(df.stringDF.DF.len)
  df.stringDF = df.stringDF.addColumn(columnName, columnData)
  df.syncIndex
  result = df

proc addColumn*[T](df: DataFrame, columnName: string, columnData: seq[T]): DataFrame=
  result = df.addColumn(columnName, columnData.toNdArray)

proc addColumn*[T](df: DataFrame, columnName: string, columnData: openArray[T]): DataFrame=
  result = df.addColumn(columnName, columnData.toNdArray)

proc removeColumn*[T](tdf: TypedDataFrame[T], colName: string) =
  var
    colIdx = tdf.getColIdx(colName)
  assert(colIdx >= 0, fmt"{colName} is not found in TypedDataFrame.")
  tdf.columns.delete(colIdx.Natural, colIdx.Natural)
  tdf.DF.delete(colIdx.Natural, colIdx.Natural)

proc removeColumn*(df: DataFrame, colName: string) =
  var
    colIdx = df.getColIdx(colName)
    colDtype = df.getColDtype(colName)
    colIdxInTypedDF = df.getColIdxInTypedDF(colName)
  assert(colIdx >= 0, fmt"{colName} is not found in TypedDataFrame.")
  df.columns.delete(colIdx.Natural, colIdx.Natural)
  df.colIdxInTypedDF.delete(colIdx.Natural, colIdx.Natural)
  df.dtypes.delete(colIdx.Natural, colIdx.Natural)
  case colDtype:
  of "int":
    df.intDF.removeColumn(colName)
  of "float":
    df.floatDF.removeColumn(colName)
  of "bool":
    df.boolDF.removeColumn(colName)
  of "string":
    df.stringDF.removeColumn(colName)
  else:
    quit(fmt"Unknown dtype {colDtype}.")

proc printRowHelper*(df: DataFrame, colIdx: int, rowIdx: int, maxColWidth: int): string=
  let
    columnName = df.columns[colIdx]
  if df.dtypes[colIdx] == "string":
    result = df.stringDF[columnName].at(rowIdx)
  elif df.dtypes[colIdx] == "int":
    result = `$`(df.intDF[columnName].at(rowIdx))
  elif df.dtypes[colIdx] == "bool":
    result = `$`(df.boolDF[columnName].at(rowIdx))
  elif df.dtypes[colIdx] == "float":
    result = `$`(df.floatDF[columnName].at(rowIdx))
  else:
    quit("?????")
  if result.len>maxColWidth:
    result = result[0 ..< maxColWidth]

proc printRow*(df: DataFrame, rowIdx: int, maxColWidth: int): string=
  var
    tmp = ""
  
  result = "" & fmt"{df.rowIndex[rowIdx]}".fixedWidthStr(maxColWidth)
  if not df.indexDF.isNil:
    for idxColName in df.indexDF.columns:
      result &= fmt"{printRowHelper(df.indexDF, df.indexDF.getColIdx(idxColName), rowIdx, maxColWidth)}".fixedWidthStr(maxColWidth)
  result[result.len-1] = '|'
  
  if df.dtypes.len > 20:
    for colIdx in 0..<10:
      tmp = df.printRowHelper(colIdx, rowIdx, maxColWidth)
      result &= fmt"{tmp}".fixedWidthStr(maxColWidth)
    result &= "...".fixedWidthStr(maxColWidth)
    for colIdx in (df.dtypes.len-10)..(df.dtypes.len-1):
      tmp = df.printRowHelper(colIdx, rowIdx, maxColWidth)
      result &= fmt"{tmp}".fixedWidthStr(maxColWidth)
  else:
    for colIdx in 0..<df.dtypes.len:
      tmp = df.printRowHelper(colIdx, rowIdx, maxColWidth)
      result &= fmt"{tmp}".fixedWidthStr(maxColWidth)
  if df.numRows > 0: result &= "\n"

proc printRow*(df: TypedDataFrame, rowIdx: int, maxColWidth: int): string=
  var
    tmp = ""
  
  result = "" & fmt"{df.rowIndex[rowIdx]}".fixedWidthStr(maxColWidth)
  if not df.indexDF.isNil:
    for idxColName in df.indexDF.columns:
      result &= fmt"{printRowHelper(df.indexDF, df.indexDF.getColIdx(idxColName), rowIdx, maxColWidth)}".fixedWidthStr(maxColWidth)
    result[result.len-1] = '|'
  
  if df.columns.len > 20:
    for colIdx in 0..<10:
      tmp = `$`(df.DF[colIdx].at(rowIdx))
      result &= fmt"{tmp}".fixedWidthStr(maxColWidth)
    result &= "...".fixedWidthStr(maxColWidth)
    for colIdx in (df.columns.len-10)..(df.columns.len-1):
      tmp = `$`(df.DF[colIdx].at(rowIdx))
      result &= fmt"{tmp}".fixedWidthStr(maxColWidth)
  else:
    for colIdx in 0..<df.columns.len:
      tmp = `$`(df.DF[colIdx].at(rowIdx))
      result &= fmt"{tmp}".fixedWidthStr(maxColWidth)
  if df.numRows > 0: result &= "\n"

proc `$`*(df: DataFrame, colWidth = 10): string=
  var
    maxColWidth = colWidth
    printCols = df.columns.len + 1
  for colName in df.columns:
    if colName.len > maxColWidth: maxColWidth = colName.len
  if printCols > 20: printCols = 21
  if not df.indexDF.isNil:
    printCols += df.indexDF.columns.len
  result = "=".repeat(maxColWidth).repeat(printCols) & "\n"
  result &= "index".fixedWidthStr(maxColWidth)
  if not df.indexDF.isNil:
    for idxColName in df.indexDF.columns:
      result &= fmt"idx:{idxColName}".fixedWidthStr(maxColWidth)
  result[result.len-1] = '|'

  if df.columns.len > 20:
    for colIdx in 0..<10: result &= fmt"{df.columns[colIdx]}".fixedWidthStr(maxColWidth)
    result &= "...".fixedWidthStr(maxColWidth)
    for colIdx in (df.columns.len-10)..(df.columns.len-1): result &= fmt"{df.columns[colIdx]}".fixedWidthStr(maxColWidth)
    result &= "\n"

    result &= "int".fixedWidthStr(maxColWidth)
    if not df.indexDF.isNil:
      for idxColName in df.indexDF.columns:
        result &= fmt"{df.indexDF.getColDtype(idxColName)}".fixedWidthStr(maxColWidth)
    result[result.len-1] = '|'

    for colIdx in 0..<10: result &= fmt"{df.dtypes[colIdx]}".fixedWidthStr(maxColWidth)
    result &= "...".fixedWidthStr(maxColWidth)
    for colIdx in (df.columns.len-10)..(df.columns.len-1): result &= fmt"{df.dtypes[colIdx]}".fixedWidthStr(maxColWidth)
    result &= "\n"
  else:
    for colIdx in 0..< df.columns.len: result &= fmt"{df.columns[colIdx]}".fixedWidthStr(maxColWidth)
    result &= "\n"

    result &= "int".fixedWidthStr(maxColWidth)
    if not df.indexDF.isNil:
      for idxColName in df.indexDF.columns:
        result &= fmt"{df.indexDF.getColDtype(idxColName)}".fixedWidthStr(maxColWidth)
    result[result.len-1] = '|'
    for colIdx in 0..< df.columns.len: result &= fmt"{df.dtypes[colIdx]}".fixedWidthStr(maxColWidth)
    result &= "\n"
  result &= "-".repeat(maxColWidth).repeat(printCols) & "\n"

  if df.numRows >= 10:
    for rowIdx in 0..<5:
      result &= df.printRow(rowIdx, maxColWidth)
    result &= "...".fixedWidthStr(maxColWidth).repeat(printCols) & "\n"
    for rowIdx in (df.numRows-5)..(df.numRows-1):
      result &= df.printRow(rowIdx, maxColWidth)
  else:
    for rowIdx in 0..<df.numRows:
      result &= df.printRow(rowIdx, maxColWidth)
  result &= "=".repeat(maxColWidth).repeat(printCols) & "\n"
  result &= fmt"({df.columns.len} columns, {df.numRows} rows)"

proc `$`*[T](df: TypedDataFrame[T], colWidth = 10): string=
  var
    printCols = df.columns.len + 1
    maxColWidth = colWidth
  for colName in df.columns:
    if colName.len > maxColWidth: maxColWidth = colName.len
  if printCols > 20: printCols = 21
  if not df.indexDF.isNil:
    printCols += df.indexDF.columns.len
  result = '_'.repeat(maxColWidth).repeat(printCols) & "\n" & fmt"Typed DataFrame : [{$T}]" & "\n"
  result &= "=".repeat(maxColWidth).repeat(printCols) & "\n"
  result &= "index".fixedWidthStr(maxColWidth)
  if not df.indexDF.isNil:
    for idxColName in df.indexDF.columns:
      result &= fmt"idx:{idxColName:}".fixedWidthStr(maxColWidth)
  result[result.len-1] = '|'

  if df.columns.len > 20:
    for colIdx in 0..<10: result &= fmt"{df.columns[colIdx]}".fixedWidthStr(maxColWidth)
    result &= fmt"...".fixedWidthStr(maxColWidth)
    for colIdx in (df.columns.len-10)..(df.columns.len-1): result &= fmt"{df.columns[colIdx]}".fixedWidthStr(maxColWidth)
    result &= "\n"
  else:
    for colIdx in 0..< df.columns.len: result &= fmt"{df.columns[colIdx]}".fixedWidthStr(maxColWidth)
    result &= "\n"
  result &= "-".repeat(maxColWidth).repeat(printCols) & "\n"

  if df.numRows >= 10:
    for rowIdx in 0..<5:
      result &= df.printRow(rowIdx, maxColWidth)
    result &= "...".fixedWidthStr(maxColWidth).repeat(printCols) & "\n"
    for rowIdx in (df.numRows-5)..(df.numRows-1):
      result &= df.printRow(rowIdx, maxColWidth)
  else:
    for rowIdx in 0..<df.numRows:
      result &= df.printRow(rowIdx, maxColWidth)
  result &= "=".repeat(maxColWidth).repeat(printCols) & "\n"
  result &= fmt"({df.columns.len} columns, {df.numRows} rows)"

proc `[]`*(df: DataFrame, colName: string): DataFrame=
  result = initDataFrame()
  let
    colIdx = df.getColIdx(colName)
    colDtype = df.getColDtype(colName)
  result.rowIndex = df.rowIndex
  result.indexDF = df.indexDF
  case colDtype:
  of "int":
    discard result.addColumn(colName, df.intDF[colName])
  of "float":
    discard result.addColumn(colName, df.floatDF[colName])
  of "bool":
    discard result.addColumn(colName, df.boolDF[colName])
  of "string":
    discard result.addColumn(colName, df.stringDF[colName])
  else:
    quit(fmt"Unknown dtype {colDtype}.")

proc `[]`*(df: DataFrame, colNames: seq[string]): DataFrame=
  result = initDataFrame()
  let
    colIdxs = newSeq[int](colNames.len)
    colDtypes = newSeq[string](colNames.len)
  result.rowIndex = df.rowIndex
  result.indexDF = df.indexDF
  for i, colName in colNames:
    var
      colDtype = df.getColDtype(colName)
    case colDtype:
    of "int":
      discard result.addColumn(colNames[i], df.intDF[colNames[i]])
    of "float":
      discard result.addColumn(colNames[i], df.floatDF[colNames[i]])
    of "bool":
      discard result.addColumn(colNames[i], df.boolDF[colNames[i]])
    of "string":
      discard result.addColumn(colNames[i], df.stringDF[colNames[i]])
    else:
      quit(fmt"Unknown dtype {colDtype}.")

proc `[]`*(df: DataFrame, colNames: openArray[string]): DataFrame=
  result = `[]`(df, colNames.toSeq)

proc `[]=`*[T](df: DataFrame, colName: string, columnData: NdArray[T]) =
  discard df.addColumn(colName, columnData)

proc `[]=`*[T](df: DataFrame, colName: string, columnData: seq[T]) =
  discard df.addColumn(colName, columnData)

proc `[]=`*[T](df: DataFrame, colName: string, columnData: openArray[T]) =
  discard df.addColumn(colName, columnData)

proc reset_index*(df: DataFrame, keep_index = true) =
  if not df.indexDF.isNil:
    for idxColName in df.indexDF.columns:
      if df.columns.contains(idxColName): continue
      var
        idxColDtype = df.indexDF.getColDtype(idxColName)
        idxColIdxInTypedDF = df.indexDF.getColIdxInTypedDF(idxColName)
      case idxColDtype:
      of "int":
        discard df.addColumn(idxColName, df.indexDF.intDF[idxColName])
      of "float":
        discard df.addColumn(idxColName, df.indexDF.floatDF[idxColName])
      of "bool":
        discard df.addColumn(idxColName, df.indexDF.boolDF[idxColName])
      of "string":
        discard df.addColumn(idxColName, df.indexDF.stringDF[idxColName])
    df.indexDF = nil
  if keep_index:
    var
      indexName = "index"
    while df.columns.contains(indexName):
      indexName &= "_"
    discard df.addColumn(indexName, df.rowIndex.toNdArray)
  df.rowIndex = xrange(df.rowIndex.len)
  df.syncIndex

proc set_index*(df: DataFrame, colName: string, keep_index = true) =
  let
    colIdx = df.getColIdx(colName)
    colDtype = df.getColDtype(colName)
    colIdxInTypedDF = df.getColIdxInTypedDF(colName)
  var
    indexDF = initDataFrame()
  indexDF.numRows = df.numRows
  indexDF.rowIndex = df.rowIndex

  df.reset_index(keep_index)
  
  case colDtype:
  of "int":
    discard indexDF.addColumn(colName, df.intDF[colName])
  of "float":
    discard indexDF.addColumn(colName, df.floatDF[colName])
  of "string":
    discard indexDF.addColumn(colName, df.stringDF[colName])
  of "bool":
    discard indexDF.addColumn(colName, df.boolDF[colName])
  else:
    quit(fmt"Unknown dtype: {colDtype}")
  df.removeColumn(colName)
  df.indexDF = indexDF
  df.syncIndex

proc set_index*(df: DataFrame, colNames: seq[string], keep_index = true) =
  var
    colIdxs = newSeq[int](colNames.len)
    colDtypes = newSeq[string](colNames.len)
    colIdxInTypedDFs = newSeq[int](colNames.len)
    indexDF = initDataFrame()
  for i, colName in colNames:
    colIdxs[i] = df.getColIdx(colName)
    colDtypes[i] = df.getColDtype(colName)
    colIdxInTypedDFs[i] = df.getColIdxInTypedDF(colName)
  indexDF.numRows = df.numRows
  indexDF.rowIndex = df.rowIndex

  df.reset_index(keep_index)
  
  for i, colDtype in colDtypes:
    case colDtype:
    of "int":
      discard indexDF.addColumn(colNames[i], df.intDF[colNames[i]])
    of "float":
      discard indexDF.addColumn(colNames[i], df.floatDF[colNames[i]])
    of "string":
      discard indexDF.addColumn(colNames[i], df.stringDF[colNames[i]])
    of "bool":
      discard indexDF.addColumn(colNames[i], df.boolDF[colNames[i]])
    else:
      quit(fmt"Unknown dtype: {colDtype}")
  
  for colName in colNames:
    df.removeColumn(colName)
  
  df.indexDF = indexDF
  df.syncIndex

proc set_index*(df: DataFrame, colNames: openArray[string], keep_index = true) =
  set_index(df, colNames.toSeq, keep_index)

when isMainModule:
  import ../common
  var
    a = arange(10)
    b = arange(10.0)
    c = xrange(10).asString
    d = initDataFrame()
  
  d["A"] = a
  d["B"] = b
  d["C"] = c
  echo d
  d.set_index(["A","B"], false)
  echo  d
  echo d.stringDF.`$`(20)
  echo d.stringDF[@["C"]]