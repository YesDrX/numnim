{.passL: "-lgsl".}
when defined(windows):
  const
    libSuffix = ".dll"
    libPrefix = ""
elif defined(macosx):
  const
    libSuffix = ".dylib"
    libPrefix = "lib"
else:
  const
    libSuffix = ".so"
    libPrefix = "lib"
const
  gsl {.strdefine.} = "gsl"
  libName = libPrefix & gsl & libSuffix

{.hint: "Using GNU/GSL library with name: " & libName .}

proc gsl_stats_mean*(arr: ptr cdouble, stride: cint, n: cint): cdouble {.importc: "gsl_stats_mean", dynlib:libName.}
proc gsl_stats_variance*(arr: ptr cdouble, stride: cint, n: cint): cdouble {.importc: "gsl_stats_variance", dynlib:libName.}