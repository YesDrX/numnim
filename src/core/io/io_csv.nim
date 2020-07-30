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

proc parseIntColumn(columnData: seq[string]): seq[int]=
  return columnData.map((x) => x.parseInt)

proc parseBoolColumn(columnData: seq[string]): seq[bool]=
  return columnData.map((x) => x.parseBool)

proc read_csv(fileName: string,
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

when isMainModule:
  var
    df1 = read_csv("/home/richard/workspace/nim/numnim/src/core/io/data.csv", dtypes = @["float","string"])
    df = read_csv("/home/richard/workspace/nim/numnim/src/core/io/data.csv")
    int_df = df.intDF
    float_df = df.floatDF
    bool_df = df.boolDF
  echo df
  # echo int_df
  # echo float_df
  echo float_df["A1"]
  # echo bindSym("float")
  # import ../ndarray/ndarray
  # import ../common
  # echo xrange(100).astype(float)
  # echo xrange(100).astype(bindSym("float").type).toNdArray(@[10,10])
  # echo 
  # echo (@[1,2],@["A","bc"])[0]

  # echo parseFloat("nan")
  # echo parseFloat("nan")
  # var
  # echo parseFloat("nan")
  #   a = @["nan", "1.", "5.0", "9."]
  #   b = @["true","false","true"]
  # echo a.parseFloatColumn
  # echo b.parseBoolColumn