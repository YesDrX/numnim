# Package

version       = "0.1.0"
author        = "YesDrX"
description   = "A numpy like ndarray and dataframe library for nim-lang."
license       = "MIT"
srcDir        = "src"


skipDirs      = @["core/parallel"]
# Dependencies
requires "nim >= 1.0.0", "nimblas >= 0.2.2", "nimlapack >= 0.2.0", "nimpy >= 0.1.0"

task test, "Run the tests":
    exec "nim c -r tests/incltests.nim"
    rmFile "tests/incltests"
