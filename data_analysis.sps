* Encoding: UTF-8.

*** SET DATA PATH

CD 'C:\ADD\PATH\TO\CSV\DATA\FILES\'.


*** IMPORT subject-level CSV data

PRESERVE.
SET DECIMAL DOT.

GET DATA  /TYPE=TXT
  /FILE="SAT_subjectLevel.csv"
  /ENCODING='UTF8'
  /DELIMITERS=","
  /ARRANGEMENT=DELIMITED
  /FIRSTCASE=2
  /DATATYPEMIN PERCENTAGE=95.0
  /VARIABLES=
  V1 2X
  ID A8
  age F3
  gender F1
  SAT_Total_points F5.0
  SAT_PER F10.2
  group F1
  IDP_CORR 4X
  IDP_MaxCORR 4X
  IDP_PER F10.2
  IDP_ERR 3X
  IDP_ACC F10.2
  IDP_RT F10.2
  IDP_RT_SD F10.2
  SWM_CORR 4X
  SWM_MaxCORR 4X
  SWM_PER F10.2
  SWM_ERR 4X
  SWM_ACC F10.2
  SWM_RT F10.2
  SWM_RT_SD F10.2
  SAT_RT F10.2
  MeanPD F10.3
  StdPDSamples F10.3
  aud_sum F2
  graduat F1
  job F1
  Raven_PER F10.2
  subject 4X
  order F3
  model_alpha F10.3
  model_beta F10.3
  model_theta F10.3
  /MAP.
RESTORE.
CACHE.
EXECUTE.
DATASET NAME SAT_subjectLevel WINDOW=FRONT.

COMPUTE HEEQ=graduat  >= 5.
EXECUTE.

ADD VALUE LABELS group 0'HC' 1'AUD'.
ADD VALUE LABELS gender 0'male' 1'female'.
ADD VALUE LABELS order 1'normal' 2'reversed'.
ADD VALUE LABELS graduat 0'Schüler/in allg.bild. Schule' 1'Schüler/in berufsorientierte S.' 2'Hauptschulabschluss' 3'Realschulabschluss' 4'Polytechnischen Oberschule' 5'Fachhochschulreife' 6'Hochschulreife/Abitur'.
ADD VALUE LABELS job 0'no' 1'yes'.
ADD VALUE LABELS HEEQ 0'no' 1'yes'.

VARIABLE LABELS 
    IDP_PER 'IDP_PER (%)' IDP_RT 'IDP_RT (s)' IDP_RT_SD 'IDP_RT_SD (s)' 
    SWM_PER 'SWM_PER (%)' SWM_RT 'SWM_RT (s)' SWM_RT_SD 'SWM_RT_SD (s)' 
    SAT_PER 'SAT_PER (%)' SAT_RT 'SAT_RT (s)' MeanPD 'Mean Planning Depth' 
    job 'Employed' graduat 'graduation level Germany' Raven_PER 'Raven_PER (%)'
    aud_sum 'SUM AUD criteria'.


SAVE OUTFILE='SAT_subjectLevel.sav'
  /COMPRESSED.


*** IMPORT condition-level CSV data
    
PRESERVE.
SET DECIMAL DOT.

GET DATA  /TYPE=TXT
  /FILE="SAT_conditionLevel.csv"
  /ENCODING='UTF8'
  /DELIMITERS=","
  /ARRANGEMENT=DELIMITED
  /FIRSTCASE=2
  /DATATYPEMIN PERCENTAGE=95.0
  /VARIABLES=
  V1 2X
  ID A8
  steps F1
  noise F1
  age F3
  gender F1
  SAT_Total_points F5.0
  SAT_PER F10.2
  group F1
  IDP_CORR 4X
  IDP_MaxCORR 4X
  IDP_PER F10.2
  IDP_ERR 3X
  IDP_ACC F10.2
  IDP_RT F10.2
  IDP_RT_SD F10.2
  SWM_CORR 4X
  SWM_MaxCORR 4X
  SWM_PER F10.2
  SWM_ERR 4X
  SWM_ACC F10.2
  SWM_RT F10.2
  SWM_RT_SD F10.2
  SAT_RT F10.2
  MeanPD F10.3
  StdPDSamples F10.3
  aud_sum F2
  graduat F1
  job F1
  Raven_PER F10.2
  subject 4X
  order F3
  model_alpha F10.3
  model_beta F10.3
  model_theta F10.3
  /MAP.
RESTORE.
CACHE.
EXECUTE.
DATASET NAME SAT_conditionLevel WINDOW=FRONT.

COMPUTE steps=steps - 2. /* dummy-code steps variable.
EXECUTE.

ADD VALUE LABELS noise 0'low noise' 1'high noise'.
ADD VALUE LABELS steps 0'2 steps' 1'3 steps'.
ADD VALUE LABELS group 0'HC' 1'AUD'.
ADD VALUE LABELS gender 0'male' 1'female'.
ADD VALUE LABELS order 1'normal' 2'reversed'.
ADD VALUE LABELS graduat 0'Schüler/in allg.bild. Schule' 1'Schüler/in berufsorientierte S.' 2'Hauptschulabschluss' 3'Realschulabschluss' 4'Polytechnischen Oberschule' 5'Fachhochschulreife' 6'Hochschulreife/Abitur'.
ADD VALUE LABELS job 0'no' 1'yes'.

VARIABLE LABELS 
    IDP_PER 'IDP_PER (%)' IDP_RT 'IDP_RT (s)' IDP_RT_SD 'IDP_RT_SD (s)' 
    SWM_PER 'SWM_PER (%)' SWM_RT 'SWM_RT (s)' SWM_RT_SD 'SWM_RT_SD (s)' 
    SAT_PER 'SAT_PER (%)' SAT_RT 'SAT_RT (s)' MeanPD 'Mean Planning Depth'
    job 'Employed' graduat 'graduation level Germany' Raven_PER 'Raven_PER (%)'
    aud_sum 'SUM AUD criteria'.

SAVE OUTFILE='SAT_conditionLevel.sav'
  /COMPRESSED.


*** IMPORT miniblock-level CSV data

PRESERVE.
SET DECIMAL DOT.

GET DATA  /TYPE=TXT
  /FILE="SAT_singleMiniblocks.csv"
  /ENCODING='UTF8'
  /DELIMITERS=","
  /ARRANGEMENT=DELIMITED
  /FIRSTCASE=2
  /DATATYPEMIN PERCENTAGE=95.0
  /VARIABLES=
  V1 2X
  ID A8
  age F3
  gender F1
  SAT_Total_points F5
  SAT_PER F10.2
  group F1
  IDP_CORR F2
  IDP_MaxCORR F2
  IDP_PER F10.2
  IDP_ERR F2
  IDP_ACC F10.2
  IDP_RT F10.2
  IDP_RT_SD F10.2
  SWM_CORR F2
  SWM_MaxCORR F2
  SWM_PER F10.2
  SWM_ERR F2
  SWM_ACC F10.2
  SWM_RT F10.2
  SWM_RT_SD F10.2
  block_num F3
  noise F1
  steps F1
  SAT_RT F10.2
  MeanPD F10.2
  StdPDSamples F10.2
  aud_sum F2
  graduat F1
  job F1
  Raven_PER F10.2
  subject 4X
  order F1
  model_alpha F10.3
  model_beta F10.3
  model_theta F10.3
  block_id F3
  /MAP.
RESTORE.
CACHE.
EXECUTE.
DATASET NAME SAT_singleMiniblocks WINDOW=FRONT.

COMPUTE steps=steps - 2. /* dummy-code steps variable.
EXECUTE.

ADD VALUE LABELS noise 0'low noise' 1'high noise'.
ADD VALUE LABELS steps 0'2 steps' 1'3 steps'.
ADD VALUE LABELS group 0'HC' 1'AUD'.
ADD VALUE LABELS gender 0'male' 1'female'.
ADD VALUE LABELS order 1'normal' 2'reversed'.
ADD VALUE LABELS graduat 0'Schüler/in allg.bild. Schule' 1'Schüler/in berufsorientierte S.' 2'Hauptschulabschluss' 3'Realschulabschluss' 4'Polytechnischen Oberschule' 5'Fachhochschulreife' 6'Hochschulreife/Abitur'.
ADD VALUE LABELS job 0'no' 1'yes'.

VARIABLE LABELS 
    IDP_PER 'IDP_PER (%)' IDP_RT 'IDP_RT (s)' IDP_RT_SD 'IDP_RT_SD (s)' 
    SWM_PER 'SWM_PER (%)' SWM_RT 'SWM_RT (s)' SWM_RT_SD 'SWM_RT_SD (s)' 
    SAT_PER 'SAT_PER (%)' SAT_RT 'SAT_RT (s)' MeanPD 'Mean Planning Depth'
    job 'Employed' graduat 'graduation level Germany' Raven_PER 'Raven_PER (%)'
    aud_sum 'SUM AUD criteria'.

SAVE OUTFILE='SAT_singleMiniblocks.sav'
  /COMPRESSED.



*** OR LOAD SPSS DATA FILES.

GET  FILE='SAT_subjectLevel.sav'.
DATASET NAME SAT_subjectLevel WINDOW=FRONT.

GET  FILE='SAT_conditionLevel.sav'.
DATASET NAME SAT_conditionLevel WINDOW=FRONT.

GET  FILE='SAT_singleMiniblocks.sav'.
DATASET NAME SAT_singleMiniblocks WINDOW=FRONT.


***EXCLUSION.
* exclude cases with beta < 1 or SAT_Total_points below 500.
DATASET ACTIVATE SAT_subjectLevel.
USE ALL.
COMPUTE excl_beta_or_points_filter = (model_beta > 1  &  SAT_Total_points >= 500).
VARIABLE LABELS excl_beta_or_points_filter 'model_beta <= 1  |  SAT_Total_points < 500 (FILTER)'.
VALUE LABELS excl_beta_or_points_filter 0 'excluded' 1 'included'.
FORMATS excl_beta_or_points_filter (f1.0).
FILTER BY excl_beta_or_points_filter.
EXECUTE.

DATASET ACTIVATE SAT_conditionLevel.
USE ALL.
COMPUTE excl_beta_or_points_filter = (model_beta > 1  &  SAT_Total_points >= 500).
VARIABLE LABELS excl_beta_or_points_filter 'model_beta <= 1  |  SAT_Total_points < 500 (FILTER)'.
VALUE LABELS excl_beta_or_points_filter 0 'excluded' 1 'included'.
FORMATS excl_beta_or_points_filter (f1.0).
FILTER BY excl_beta_or_points_filter.
EXECUTE.

DATASET ACTIVATE SAT_singleMiniblocks.
USE ALL.
COMPUTE excl_beta_or_points_filter = (model_beta > 1  &  SAT_Total_points >= 500).
VARIABLE LABELS excl_beta_or_points_filter 'model_beta <= 1  |  SAT_Total_points < 500 (FILTER)'.
VALUE LABELS excl_beta_or_points_filter 0 'excluded' 1 'included'.
FORMATS excl_beta_or_points_filter (f1.0).
FILTER BY excl_beta_or_points_filter.
EXECUTE.



***ANALYSIS.

*** group comparison gender.

DATASET ACTIVATE SAT_subjectLevel.
CROSSTABS
  /TABLES=group BY gender
  /FORMAT=AVALUE TABLES
  /STATISTICS=CHISQ 
  /CELLS=COUNT
  /COUNT ROUND CELL.

*** group comparison employment.

DATASET ACTIVATE SAT_subjectLevel.
CROSSTABS
  /TABLES=group BY job
  /FORMAT=AVALUE TABLES
  /STATISTICS=CHISQ 
  /CELLS=COUNT
  /COUNT ROUND CELL.

*** group comparison higher education entrance qualification.

DATASET ACTIVATE SAT_subjectLevel.
CROSSTABS
  /TABLES=group BY HEEQ
  /FORMAT=AVALUE TABLES
  /STATISTICS=CHISQ 
  /CELLS=COUNT
  /COUNT ROUND CELL.

*** group comparisons normality test.

DATASET ACTIVATE SAT_subjectLevel.
EXAMINE VARIABLES=age job graduat SAT_PER SAT_RT MeanPD model_alpha model_beta model_theta SWM_PER SWM_RT IDP_PER IDP_RT Raven_PER BY group
  /PLOT HISTOGRAM NPPLOT
  /COMPARE GROUPS
  /STATISTICS NONE
  /CINTERVAL 95
  /MISSING PAIRWISE
  /NOTOTAL.

*** group comparisons standard t-test.

DATASET ACTIVATE SAT_subjectLevel.
T-TEST groupS=group(0 1)
  /MISSING=ANALYSIS
  /VARIABLES=age job graduat SAT_PER SAT_RT MeanPD model_alpha model_beta model_theta SWM_PER SWM_RT IDP_PER IDP_RT Raven_PER
  /ES DISPLAY(TRUE)
  /CRITERIA=CI(.95).

*** group comparisons nonparametric test
    
NPAR TESTS
  /M-W= age job graduat SAT_PER SAT_RT MeanPD model_alpha model_beta model_theta SWM_PER SWM_RT IDP_PER IDP_RT Raven_PER BY group(0 1)
  /MISSING ANALYSIS.



*** Correlations.

DATASET ACTIVATE SAT_subjectLevel.
CORRELATIONS
  /VARIABLES=age job graduat SAT_PER SAT_RT MeanPD model_alpha model_beta model_theta SWM_PER SWM_RT IDP_PER IDP_RT Raven_PER
  /PRINT=TWOTAIL NOSIG FULL
  /MISSING=PAIRWISE.




*** LME model of MeanPD incl. group*steps interaction.

DATASET ACTIVATE SAT_singleMiniblocks.
MIXED MeanPD WITH group steps
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(100) SCORING(10) 
    SINGULAR(0.0000000001) HCONVERGE(0.001, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.00001, ABSOLUTE)
  /FIXED=group steps group*steps | SSTYPE(3)
  /METHOD=ML
  /PRINT=CORB  SOLUTION TESTCOV
  /RANDOM=INTERCEPT steps | SUBJECT(ID) COVTYPE(VC)
  /SAVE = PRED(LME_PRED) RESID(LME_RESID).
FORMATS LME_PRED(F10.2) LME_RESID(F10.2). 


*** PLOT: lme model residuals.

DATASET ACTIVATE SAT_singleMiniblocks.
GGRAPH
  /GRAPHDATASET NAME="graphdataset" VARIABLES=LME_PRED LME_RESID MISSING=LISTWISE REPORTMISSING=NO
  /GRAPHSPEC SOURCE=INLINE
  /FITLINE TOTAL=NO SUBgroup=NO
  /FRAME OUTER=NO INNER=NO
  /GRIDLINES XAXIS=NO YAXIS=NO
  /STYLE GRADIENT=NO.
BEGIN GPL
  SOURCE: s=userSource(id("graphdataset"))
  DATA: LME_PRED=col(source(s), name("LME_PRED"))
  DATA: LME_RESID=col(source(s), name("LME_RESID"))
  GUIDE: axis(dim(1), label("Predicted Values"))
  GUIDE: axis(dim(2), label("Residuals"))
  GUIDE: form.line(position(*,0))
  ELEMENT: point(position(LME_PRED*LME_RESID))
END GPL.

*** PLOT: lme model historgram residuals.

DATASET ACTIVATE SAT_singleMiniblocks.
GGRAPH
  /GRAPHDATASET NAME="graphdataset" VARIABLES=LME_RESID MISSING=LISTWISE REPORTMISSING=NO
  /GRAPHSPEC SOURCE=INLINE
  /FRAME OUTER=NO INNER=NO
  /GRIDLINES XAXIS=NO YAXIS=NO
  /STYLE GRADIENT=NO.
BEGIN GPL
  SOURCE: s=userSource(id("graphdataset"))
  DATA: LME_RESID=col(source(s), name("LME_RESID"))
  GUIDE: axis(dim(1), label("Residuals"))
  GUIDE: axis(dim(2), label("Frequency"))
  ELEMENT: interval(position(summary.count(bin.rect(LME_RESID))), shape.interior(shape.square))
  ELEMENT: line(position(density.normal(LME_RESID)))
END GPL.

*** PLOT: lme model Q-Q plot.

DATASET ACTIVATE SAT_singleMiniblocks.
PPLOT
  /VARIABLES=LME_RESID
  /NOLOG
  /NOSTANDARDIZE
  /TYPE=Q-Q
  /FRACTION=BLOM
  /TIES=MEAN
  /DIST=NORMAL.








*** LME model of MeanPD incl. group*steps interaction and SWM_PER and SAT_RT.

DATASET ACTIVATE SAT_singleMiniblocks.
MIXED MeanPD WITH group steps SWM_PER SAT_RT
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(100) MXSTEP(100) SCORING(10) 
    SINGULAR(0.0000000001) HCONVERGE(0.001, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.00001, ABSOLUTE)
  /FIXED=group steps group*steps SWM_PER SAT_RT  | SSTYPE(3)
  /METHOD=ML
  /PRINT=CORB  SOLUTION TESTCOV
  /RANDOM=INTERCEPT steps | SUBJECT(ID) COVTYPE(VC)
  /SAVE = PRED(LME_PRED2) RESID(LME_RESID2).
FORMATS LME_PRED2(F10.2) LME_RESID2(F10.2). 


*** PLOT: lme model residuals.

DATASET ACTIVATE SAT_singleMiniblocks.
GGRAPH
  /GRAPHDATASET NAME="graphdataset" VARIABLES=LME_PRED2 LME_RESID2 MISSING=LISTWISE REPORTMISSING=NO
  /GRAPHSPEC SOURCE=INLINE
  /FITLINE TOTAL=NO SUBgroup=NO
  /FRAME OUTER=NO INNER=NO
  /GRIDLINES XAXIS=NO YAXIS=NO
  /STYLE GRADIENT=NO.
BEGIN GPL
  SOURCE: s=userSource(id("graphdataset"))
  DATA: LME_PRED2=col(source(s), name("LME_PRED2"))
  DATA: LME_RESID2=col(source(s), name("LME_RESID2"))
  GUIDE: axis(dim(1), label("Predicted Values"))
  GUIDE: axis(dim(2), label("Residuals"))
  GUIDE: form.line(position(*,0))
  ELEMENT: point(position(LME_PRED2*LME_RESID2))
END GPL.

*** PLOT: lme model historgram residuals.

DATASET ACTIVATE SAT_singleMiniblocks.
GGRAPH
  /GRAPHDATASET NAME="graphdataset" VARIABLES=LME_RESID2 MISSING=LISTWISE REPORTMISSING=NO
  /GRAPHSPEC SOURCE=INLINE
  /FRAME OUTER=NO INNER=NO
  /GRIDLINES XAXIS=NO YAXIS=NO
  /STYLE GRADIENT=NO.
BEGIN GPL
  SOURCE: s=userSource(id("graphdataset"))
  DATA: LME_RESID2=col(source(s), name("LME_RESID2"))
  GUIDE: axis(dim(1), label("Residuals"))
  GUIDE: axis(dim(2), label("Frequency"))
  ELEMENT: interval(position(summary.count(bin.rect(LME_RESID2))), shape.interior(shape.square))
  ELEMENT: line(position(density.normal(LME_RESID2)))
END GPL.

*** PLOT: lme model Q-Q plot.

DATASET ACTIVATE SAT_singleMiniblocks.
PPLOT
  /VARIABLES=LME_RESID2
  /NOLOG
  /NOSTANDARDIZE
  /TYPE=Q-Q
  /FRACTION=BLOM
  /TIES=MEAN
  /DIST=NORMAL.








*** linear regression MeanPD covariates.

DATASET ACTIVATE SAT_subjectLevel.
REGRESSION
  /DESCRIPTIVES MEAN STDDEV CORR SIG N
  /MISSING LISTWISE
  /STATISTICS COEFF OUTS CI(95) R ANOVA COLLIN TOL CHANGE ZPP
  /CRITERIA=PIN(.05) POUT(.10)
  /NOORIGIN 
  /DEPENDENT MeanPD
  /METHOD=BACKWARD group SAT_RT  IDP_PER SWM_PER Raven_PER
  /SAVE = PRED(LMC_PRED) RESID(LMC_RESID).
FORMATS LMC_PRED(F10.2) LMC_RESID(F10.2).

*** PLOT: MeanPD linear regression model residuals.

DATASET ACTIVATE SAT_subjectLevel.
GGRAPH
  /GRAPHDATASET NAME="graphdataset" VARIABLES=LMC_PRED LMC_RESID MISSING=LISTWISE REPORTMISSING=NO
  /GRAPHSPEC SOURCE=INLINE
  /FITLINE TOTAL=NO SUBgroup=NO
  /FRAME OUTER=NO INNER=NO
  /GRIDLINES XAXIS=NO YAXIS=NO
  /STYLE GRADIENT=NO.
BEGIN GPL
  SOURCE: s=userSource(id("graphdataset"))
  DATA: LMC_PRED=col(source(s), name("LMC_PRED"))
  DATA: LMC_RESID=col(source(s), name("LMC_RESID"))
  GUIDE: axis(dim(1), label("Predicted Values"))
  GUIDE: axis(dim(2), label("Residuals"))
  GUIDE: form.line(position(*,0))
  ELEMENT: point(position(LMC_PRED*LMC_RESID))
END GPL.

*** PLOT: MeanPD linear regression model historgram residuals.

DATASET ACTIVATE SAT_subjectLevel.
GGRAPH
  /GRAPHDATASET NAME="graphdataset" VARIABLES=LMC_RESID MISSING=LISTWISE REPORTMISSING=NO
  /GRAPHSPEC SOURCE=INLINE
  /FRAME OUTER=NO INNER=NO
  /GRIDLINES XAXIS=NO YAXIS=NO
  /STYLE GRADIENT=NO.
BEGIN GPL
  SOURCE: s=userSource(id("graphdataset"))
  DATA: LMC_RESID=col(source(s), name("LMC_RESID"))
  GUIDE: axis(dim(1), label("Residuals"))
  GUIDE: axis(dim(2), label("Frequency"))
  ELEMENT: interval(position(summary.count(bin.rect(LMC_RESID))), shape.interior(shape.square))
  ELEMENT: line(position(density.normal(LMC_RESID)))
END GPL.

*** PLOT: MeanPD linear regression model Q-Q plot.

DATASET ACTIVATE SAT_subjectLevel.
PPLOT
  /VARIABLES=LMC_RESID
  /NOLOG
  /NOSTANDARDIZE
  /TYPE=Q-Q
  /FRACTION=BLOM
  /TIES=MEAN
  /DIST=NORMAL.





*** model quality linear regression.

DATASET ACTIVATE SAT_subjectLevel.
REGRESSION
  /DESCRIPTIVES MEAN STDDEV CORR SIG N
  /MISSING LISTWISE
  /STATISTICS COEFF OUTS CI(95) R ANOVA COLLIN TOL CHANGE ZPP
  /CRITERIA=PIN(.05) POUT(.10)
  /NOORIGIN 
  /DEPENDENT SAT_PER
  /METHOD=ENTER model_alpha model_beta model_theta MeanPD
    /SAVE = PRED(LMQ_PRED) RESID(LMQ_RESID).
FORMATS LMQ_PRED(F10.2) LMQ_RESID(F10.2).

*** PLOT: model quality linear regression model residuals.

DATASET ACTIVATE SAT_subjectLevel.
GGRAPH
  /GRAPHDATASET NAME="graphdataset" VARIABLES=LMQ_PRED LMQ_RESID MISSING=LISTWISE REPORTMISSING=NO
  /GRAPHSPEC SOURCE=INLINE
  /FITLINE TOTAL=NO SUBgroup=NO
  /FRAME OUTER=NO INNER=NO
  /GRIDLINES XAXIS=NO YAXIS=NO
  /STYLE GRADIENT=NO.
BEGIN GPL
  SOURCE: s=userSource(id("graphdataset"))
  DATA: LMQ_PRED=col(source(s), name("LMQ_PRED"))
  DATA: LMQ_RESID=col(source(s), name("LMQ_RESID"))
  GUIDE: axis(dim(1), label("Predicted Values"))
  GUIDE: axis(dim(2), label("Residuals"))
  GUIDE: form.line(position(*,0))
  ELEMENT: point(position(LMQ_PRED*LMQ_RESID))
END GPL.

*** PLOT: model quality linear regression historgram residuals.

DATASET ACTIVATE SAT_subjectLevel.
GGRAPH
  /GRAPHDATASET NAME="graphdataset" VARIABLES=LMQ_RESID MISSING=LISTWISE REPORTMISSING=NO
  /GRAPHSPEC SOURCE=INLINE
  /FRAME OUTER=NO INNER=NO
  /GRIDLINES XAXIS=NO YAXIS=NO
  /STYLE GRADIENT=NO.
BEGIN GPL
  SOURCE: s=userSource(id("graphdataset"))
  DATA: LMQ_RESID=col(source(s), name("LMQ_RESID"))
  GUIDE: axis(dim(1), label("Residuals"))
  GUIDE: axis(dim(2), label("Frequency"))
  ELEMENT: interval(position(summary.count(bin.rect(LMQ_RESID))), shape.interior(shape.square))
  ELEMENT: line(position(density.normal(LMQ_RESID)))
END GPL.

*** PLOT: model quality linear regression model Q-Q plot.

DATASET ACTIVATE SAT_subjectLevel.
PPLOT
  /VARIABLES=LMQ_RESID
  /NOLOG
  /NOSTANDARDIZE
  /TYPE=Q-Q
  /FRACTION=BLOM
  /TIES=MEAN
  /DIST=NORMAL.


*** linear mixed effects model 1.3 of planning depth with predictors group, Q_abs_max and group-by-Q_abs_max interaction term
    
MIXED MeanPD WITH group Q_abs_max
  /CRITERIA=DFMETHOD(SATTERTHWAITE) CIN(95) MXITER(1000) MXSTEP(10) SCORING(1) 
    SINGULAR(0.000000000001) HCONVERGE(0, ABSOLUTE) LCONVERGE(0, ABSOLUTE) PCONVERGE(0.000001, ABSOLUTE)    
  /FIXED=group Q_abs_max group*Q_abs_max | SSTYPE(3)
  /METHOD=ML
  /PRINT=CPS CORB COVB DESCRIPTIVES G  LMATRIX R SOLUTION TESTCOV
  /RANDOM=INTERCEPT Q_abs_max | SUBJECT(ID) COVTYPE(VC)
  /SAVE=FIXPRED PRED.
