import strformat
import strutils
import sequtils
import parsecsv
import streams
import sugar
import ../ndarray/ndarray
import ../dataframe/dataframe_object

proc guess_type(input: string): string=
  var
    tmpInt: int
    tmpBool: bool
    tmpFloat: float
  
  if input == "":
    return "UNKNOWN"
  if input.find('\"') >= 0 or input.find('\'') >= 0:
    return "string"
  if input.find('.') >= 0:
    try:
      tmpFloat = parseFloat(input)
      return "float"
    except:
      discard 0
    return "string"
  elif input.toUpper.find('E') >= 0:
    try:
      tmpBool = parseBool(input)
      return "bool"
    except:
      discard 0  
    return "string"
  else:
    try:
      tmpInt = parseInt(input)
      return "int"
    except:
      discard 0
    return "string"

proc update_schema(current_schema: var seq[string], current_row: seq[string]): bool=
  result = true
  if current_schema.len == 0:
    for i in 0 ..< current_row.len:
      current_schema.add("UNKNOWN")
  for i, item in current_row:
    if current_schema[i] == "UNKNOWN":
      current_schema[i] = guess_type(item)
    if current_schema[i] == "UNKNOWN": result = false

proc rowsToColumns(rows: seq[seq[string]], schema: seq[string]): seq[seq[string]] =
  assert rows.len>0
  var
    numColumns = rows[0].len
    numRows = rows.len
  result = newSeq[seq[string]](numColumns)
  for colIdx in 0 ..< numColumns:
    for rowIdx in 0..< numRows:
      if unlikely(rows[rowIdx][colIdx] == ""):
        case schema[colIdx]:
        of "int":
          result[colIdx].add("–9223372036854775808")
        of "float":
          result[colIdx].add("nan")
        of "bool":
          result[colIdx].add("false")
        of "string":
          result[colIdx].add("")
      else:
        result[colIdx].add(rows[rowIdx][colIdx])

proc parseFloatColumn(columnData: seq[string]): seq[float]=
  return columnData.map((x) => x.parseFloat)

# proc myParseInt(input: string): int=
#   try:
#     return input.parseInt
#   except:

proc parseIntColumn(columnData: seq[string]): seq[int]=
  result = newSeq[int](columnData.len)
  for idx in 0 ..< columnData.len:
    try:
      result[idx] = parseInt(columnData[idx])
    except:
      result[idx] = int32.low.int
  # return columnData.map( proc (x : string): int = (if x != int.low.`$` : x.parseInt else : int32.low.int))

proc parseBoolColumn(columnData: seq[string]): seq[bool]=
  return columnData.map((x) => x.parseBool)

proc read_csv*(fileName: string,
               header: int = 0,
               separator: char = ',',
               quoateChar: char = '\"',
               escape: char = '\x00',
               maxRowsToGuessSchema: int = 10,
               dtypes: seq[string] = @[]): DataFrame =
  let
    stream = newFileStream(fileName, fmRead)
  
  var
    headerRow, currentRow: seq[string]
    unparsedRowsBuffer: seq[seq[string]] = @[]
    unparsedColumnsBuffer: seq[seq[string]]
    dataframe: DataFrame
    toLoadHeader: bool
    columns: int
    parser: CsvParser
    type_schema: seq[string] = @[]
    type_schema_all_parsed: bool = false

  if stream.isNil:
    quit(fmt"cannot open file {fileName}.")
  
  for skip_line_no in 0 .. header - 1:
    if stream.atEnd:
      quit(fmt"skiped {skip_line_no+1} rows, but we are at EOF.")
    discard stream.readLine
  
  parser.open(stream, fileName, separator, quoateChar, escape, false)
  parser.readHeaderRow
  headerRow = parser.row
  columns = headerRow.len
  
  if dtypes.len > 0:
    for i in 0 ..< dtypes.len:
      assert(dtypes[i] in @["int","bool","float","string"], fmt"{dtypes[i]} is not a valid dtype of int/bool/string/float.")
    if dtypes.len < columns:
      type_schema = dtypes & @["UNKNOWN"].cycle(columns - dtypes.len)
    else:
      type_schema = dtypes

  while parser.readRow:
    currentRow = parser.row
    if not likely(type_schema_all_parsed):
      if update_schema(type_schema, currentRow):
        type_schema_all_parsed = true
      if parser.processedRows >= maxRowsToGuessSchema:
        type_schema_all_parsed = true
        for i in 0..<type_schema.len:
          if type_schema[i] == "UNKNOWN":
            type_schema[i] = "string"
    unparsedRowsBuffer.add(currentRow)
  unparsedColumnsBuffer = rowsToColumns(unparsedRowsBuffer, type_schema)
  dataframe = initDataFrame()
  for colIdx in 0 ..< type_schema.len:
    case type_schema[colIdx]:
    of "string":
      dataframe = dataframe.addColumn(headerRow[colIdx],unparsedColumnsBuffer[colIdx])
    of "float":
      dataframe = dataframe.addColumn(headerRow[colIdx],unparsedColumnsBuffer[colIdx].parseFloatColumn)
    of "int":
      dataframe = dataframe.addColumn(headerRow[colIdx],unparsedColumnsBuffer[colIdx].parseIntColumn)
    of "bool":
      dataframe = dataframe.addColumn(headerRow[colIdx],unparsedColumnsBuffer[colIdx].parseBoolColumn)

  parser.close
  stream.close
  result = dataframe

proc to_csv*(df: DataFrame, fileName: string, index = true, header = true) =
  var
    fileNameSaveTo = fileName
    headerRow = ""
    rowText = ""
  if not fileName.endsWith(".csv"): fileNameSaveTo &= ".csv"
  
  #output file
  let outputFile = open(fileName, fmWrite)
  defer: outputFile.close()

  #header
  if header:
    if index:
      headerRow = "index,"
      if not df.indexDF.isNil:
        for idxColName in df.indexDF.columns: headerRow &= idxColName & ","
    for colName in df.columns:
      headerRow &= colName & ","
    headerRow.delete((headerRow.len-1).Natural,(headerRow.len-1).Natural)
    outputFile.writeLine(headerRow)
  
  #rows
  for rowIdx in 0 ..< df.numRows:
    rowText = ""
    if index:
      rowText &= df.rowIndex[rowIdx].`$` & ','
      if not df.indexDF.isNil:
        for idxColIdx, idxColName in df.indexDF.columns:
          rowText &= printRowHelper(df.indexDF, idxColIdx, rowIdx, int.high) & ','
    for colIdx, colName in df.columns:
      rowText &= printRowHelper(df, colIdx, rowIdx, int.high) & ','
    rowText.delete((rowText.len-1).Natural,(rowText.len-1).Natural)
    outputFile.writeLine(rowText)

when isMainModule:
  import os
  var df : DataFrame
  
  df = read_csv(absolutePath("nim/numnim/tests/sample_data/csv/contacts.csv"))
  echo df.intdf
  echo df.intDF["id"]