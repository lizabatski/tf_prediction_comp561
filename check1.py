import pyBigWig

bw = pyBigWig.open("/scratch/ekourb/wigconvert/hg19.MGW.2nd.chr1.bw")
vals = bw.values("chr1", 0, 100)   # first 100 bases
print(vals)