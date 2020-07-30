import ../src/numnim

when isMainModule:
  var
    a = zeros(@[10,10])
    b = normal(@[10,10])
  
  echo b
  echo b.cholesky