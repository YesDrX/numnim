import ../src/numnim
when isMainModule:
  import sequtils
  import strformat
  var
    a = normal(@[2,3])
    b = normal(@[3,2])
  echo a
  echo b
  echo a.dot(b)
  echo a * b.transpose
  echo a + b.transpose