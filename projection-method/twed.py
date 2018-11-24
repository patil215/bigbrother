import os, math
from collections import defaultdict
#
# Filename: twed.c
# source code for the Time Warp Edit Distance in ANSI C.
# Author: Pierre-Francois Marteau
# Version: V1.2.a du 25/08/2014, radix addition line 101, thanks to Benjamin Herwig from University of Kassel, Germany
# Licence: GPL
# *****************************************************************
# This software and description is free delivered "AS IS" with no 
# guaranties for work at all. Its up to you testing it modify it as 
# you like, but no help could be expected from me due to lag of time 
# at the moment. I will answer short relevant questions and help as 
# my time allow it. I have tested it played with it and found no 
# problems in stability or malfunctions so far. 
# Have fun.
# ****************************************************************
# Please cite as:
# @article{Marteau:2009:TWED,
 # author = {Marteau, Pierre-Francois},
 # title = {Time Warp Edit Distance with Stiffness Adjustment for Time Series Matching},
 # journal = {IEEE Trans. Pattern Anal. Mach. Intell.},
 # issue_date = {February 2009},
 # volume = {31},
 # number = {2},
 # month = feb,
 # year = {2009},
 # issn = {0162-8828},
 # pages = {306--318},
 # numpages = {13},
 # url = {http://dx.doi.org/10.1109/TPAMI.2008.76},
 # doi = {10.1109/TPAMI.2008.76},
 # acmid = {1496043},
 # publisher = {IEEE Computer Society},
 # address = {Washington, DC, USA},
 # keywords = {Dynamic programming, Pattern recognition, Pattern recognition, time series, algorithms, similarity measures., Similarity measures, algorithms, similarity measures., time series},
# } 
# 

# 
#INPUTS
#int dim: dimension of the multivariate time series (dim=1 if scalar time series)
#double **ta: array containing the first time series; ta[i] is an array of size dim containing the multidimensional i^th sample for i in {0, .., la-1}
#int la: length of the first time series
#double *tsa: array containing the time stamps for time series ta; tsa[i] is the time stamp for sample ta[i]. The length of tsb array is expected to be lb.
#double **tb: array containing the second time series; tb[i] is an array of size dim containing the multidimensional j^th sample for j in {0, .., lb-1}
#int lb: length of the second time series
#double *tsb: array containing the time stamps for time series tb; tsb[j] is the time stamp for sample tb[j]. The length of tsb array is expected to be lb.
#double nu: value for parameter nu
#double lambda: value for parameter lambda
#int degree: power degree for the evaluation of the local distance between samples: degree>0 required
#OUTPUT
#double: the TWED distance between time series ta and tb.
##
def DTWEDL1d(dim, ta, tsa, tb, tsb, nu, lambd, degree):
    r = len(ta)
    c = len(tb)
    (dist, disti1, distj1) = (0.0, 0.0, 0.0)
    (i, j, k) = (0, 0, 0)

    # allocations
    D = defaultdict(lambda: defaultdict(float))
    Di1 = defaultdict(float)
    Dj1 = defaultdict(float)

    # local costs initializations
    for j in range(1, c + 1):
        distj1 = 0.0
        for k in range(0, dim):
            if j > 1:
                distj1 += math.pow(math.fabs(tb[j - 2][k] - tb[j - 1][k]), degree)
            else:
                distj1 += math.pow(math.fabs(tb[j - 1][k]), degree)
        Dj1[j] = distj1

    for i in range(1, r + 1):
        disti1 = 0.0
        for k in range(0, dim):
            if i > 1:
                disti1 += math.pow(math.fabs(ta[i - 2][k] - ta[i - 1][k]), degree)
            else:
                disti1 += math.pow(math.fabs(ta[i - 1][k]), degree)
        Di1[i] = disti1

        for j in range(1, c + 1):
            dist = 0.0
            for k in range(0, dim):
                dist += math.pow(math.fabs(ta[i - 1][k] - tb[j - 1][k]), degree)
                if i > 1 and j > 1:
                    dist += math.pow(math.fabs(ta[i - 2][k] - tb[j - 2][k]), degree)

            D[i][j] = math.pow(dist, 1.0 / degree)
    
    # border of the cost matrix initialization
    D[0][0] = 0.0
    for i in range(1, r + 1):
        D[i][0] = D[i - 1][0] + Di1[i]
    for j in range(1, c + 1):
        D[0][j] = D[0][j - 1] + Dj1[j]

    (dmin, htrans, dist0) = (0.0, 0.0, 0.0)
    iback = 0

    for i in range(1, r + 1):
        for j in range(1, c + 1):
            htrans = math.fabs(tsa[i - 1] - tsb[j - 1])

            if j > 1 and i > 1:
                htrans += math.fabs(tsa[i - 2] - tsb[j - 2])

            dist0 = D[i - 1][j - 1] + nu * htrans + D[i][j]
            dmin = dist0

            if i > 1:
                htrans = tsa[i - 1] - tsa[i - 2]
            else:
                htrans = tsa[i - 1]

            dist = Di1[i] + D[i - 1][j] + lambd + nu * htrans
            if dmin > dist:
                dmin = dist

            if j > 1:
                htrans = tsb[j - 1] - tsb[j - 2]
            else:
                htrans = tsb[j - 1]
            
            dist = Dj1[j] + D[i][j - 1] + lambd + nu * htrans
            if dmin > dist:
                dmin = dist

            D[i][j] = dmin

    return D[r][c]
