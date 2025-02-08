BEGIN {
  FS=","
  OFS=","
  RS="\n"
  ORS="\n"

  #A = 0
  #C = 0
  #G = 0
  #T = 0
}

{
  if (NR == 1) {
    print($0, ",A1, C1, G1, T1, A2, C2, G2, T2")
  } else {

  A = gsub( /A/,"A",$8)
  C = gsub( /C/,"C",$8)
  G = gsub( /G/,"G",$8)
  T = gsub( /T/,"T",$8)


  A2 = gsub( /A/,"A",$9)
  C2 = gsub( /C/,"C",$9)
  G2 = gsub( /G/,"G",$9)
  T2 = gsub( /T/,"T",$9)

  print($0, A, C, G, T, A2, C2, G2, T2)
}
}