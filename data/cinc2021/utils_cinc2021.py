"""
Adapted from: torch_ecg
Sources: https://github.com/DeepPSP/torch_ecg/blob/85559f873f3fbe66b91ae9261bdd2cb8e51f53a2/torch_ecg/databases/aux_data/cinc2021_aux_data.py
         https://github.com/DeepPSP/cinc2021/blob/594cbd37fdf95766f3752e90ad2ed5c9040e37f0/utils/scoring_metrics.py
"""

import os
from io import StringIO
from typing import Union, Optional, List, Sequence, Dict, Tuple
from numbers import Real

import numpy as np
import pandas as pd
from easydict import EasyDict as ED
from functools import reduce

df_weights = pd.read_csv(
    StringIO(
        """,164889003,164890007,6374002,426627000,733534002|164909002,713427006|59118001,270492004,713426002,39732003,445118002,164947007,251146004,111975006,698252002,426783006,284470004|63593006,10370003,365413008,427172004|17338001,164917005,47665007,427393009,426177001,427084000,164934002,59931005
164889003,1.0,0.5,0.475,0.3,0.475,0.4,0.3,0.3,0.35,0.35,0.3,0.425,0.45,0.35,0.25,0.3375,0.375,0.425,0.375,0.4,0.35,0.3,0.3,0.375,0.5,0.5
164890007,0.5,1.0,0.475,0.3,0.475,0.4,0.3,0.3,0.35,0.35,0.3,0.425,0.45,0.35,0.25,0.3375,0.375,0.425,0.375,0.4,0.35,0.3,0.3,0.375,0.5,0.5
6374002,0.475,0.475,1.0,0.325,0.475,0.425,0.325,0.325,0.375,0.375,0.325,0.45,0.475,0.375,0.275,0.3625,0.4,0.45,0.4,0.375,0.375,0.325,0.325,0.4,0.475,0.475
426627000,0.3,0.3,0.325,1.0,0.325,0.4,0.5,0.5,0.45,0.45,0.5,0.375,0.35,0.45,0.45,0.4625,0.425,0.375,0.425,0.2,0.45,0.5,0.5,0.425,0.3,0.3
733534002|164909002,0.475,0.475,0.475,0.325,1.0,0.425,0.325,0.325,0.375,0.375,0.325,0.45,0.475,0.375,0.275,0.3625,0.4,0.45,0.4,0.375,0.375,0.325,0.325,0.4,0.475,0.475
713427006|59118001,0.4,0.4,0.425,0.4,0.425,1.0,0.4,0.4,0.45,0.45,0.4,0.475,0.45,0.45,0.35,0.4375,0.475,0.475,0.475,0.3,0.45,0.4,0.4,0.475,0.4,0.4
270492004,0.3,0.3,0.325,0.5,0.325,0.4,1.0,0.5,0.45,0.45,0.5,0.375,0.35,0.45,0.45,0.4625,0.425,0.375,0.425,0.2,0.45,0.5,0.5,0.425,0.3,0.3
713426002,0.3,0.3,0.325,0.5,0.325,0.4,0.5,1.0,0.45,0.45,0.5,0.375,0.35,0.45,0.45,0.4625,0.425,0.375,0.425,0.2,0.45,0.5,0.5,0.425,0.3,0.3
39732003,0.35,0.35,0.375,0.45,0.375,0.45,0.45,0.45,1.0,0.5,0.45,0.425,0.4,0.5,0.4,0.4875,0.475,0.425,0.475,0.25,0.5,0.45,0.45,0.475,0.35,0.35
445118002,0.35,0.35,0.375,0.45,0.375,0.45,0.45,0.45,0.5,1.0,0.45,0.425,0.4,0.5,0.4,0.4875,0.475,0.425,0.475,0.25,0.5,0.45,0.45,0.475,0.35,0.35
164947007,0.3,0.3,0.325,0.5,0.325,0.4,0.5,0.5,0.45,0.45,1.0,0.375,0.35,0.45,0.45,0.4625,0.425,0.375,0.425,0.2,0.45,0.5,0.5,0.425,0.3,0.3
251146004,0.425,0.425,0.45,0.375,0.45,0.475,0.375,0.375,0.425,0.425,0.375,1.0,0.475,0.425,0.325,0.4125,0.45,0.475,0.45,0.325,0.425,0.375,0.375,0.45,0.425,0.425
111975006,0.45,0.45,0.475,0.35,0.475,0.45,0.35,0.35,0.4,0.4,0.35,0.475,1.0,0.4,0.3,0.3875,0.425,0.475,0.425,0.35,0.4,0.35,0.35,0.425,0.45,0.45
698252002,0.35,0.35,0.375,0.45,0.375,0.45,0.45,0.45,0.5,0.5,0.45,0.425,0.4,1.0,0.4,0.4875,0.475,0.425,0.475,0.25,0.5,0.45,0.45,0.475,0.35,0.35
426783006,0.25,0.25,0.275,0.45,0.275,0.35,0.45,0.45,0.4,0.4,0.45,0.325,0.3,0.4,1.0,0.4125,0.375,0.325,0.375,0.15,0.4,0.45,0.45,0.375,0.25,0.25
284470004|63593006,0.3375,0.3375,0.3625,0.4625,0.3625,0.4375,0.4625,0.4625,0.4875,0.4875,0.4625,0.4125,0.3875,0.4875,0.4125,1.0,0.4625,0.4125,0.4625,0.2375,0.4875,0.4625,0.4625,0.4625,0.3375,0.3375
10370003,0.375,0.375,0.4,0.425,0.4,0.475,0.425,0.425,0.475,0.475,0.425,0.45,0.425,0.475,0.375,0.4625,1.0,0.45,0.5,0.275,0.475,0.425,0.425,0.5,0.375,0.375
365413008,0.425,0.425,0.45,0.375,0.45,0.475,0.375,0.375,0.425,0.425,0.375,0.475,0.475,0.425,0.325,0.4125,0.45,1.0,0.45,0.325,0.425,0.375,0.375,0.45,0.425,0.425
427172004|17338001,0.375,0.375,0.4,0.425,0.4,0.475,0.425,0.425,0.475,0.475,0.425,0.45,0.425,0.475,0.375,0.4625,0.5,0.45,1.0,0.275,0.475,0.425,0.425,0.5,0.375,0.375
164917005,0.4,0.4,0.375,0.2,0.375,0.3,0.2,0.2,0.25,0.25,0.2,0.325,0.35,0.25,0.15,0.2375,0.275,0.325,0.275,1.0,0.25,0.2,0.2,0.275,0.4,0.4
47665007,0.35,0.35,0.375,0.45,0.375,0.45,0.45,0.45,0.5,0.5,0.45,0.425,0.4,0.5,0.4,0.4875,0.475,0.425,0.475,0.25,1.0,0.45,0.45,0.475,0.35,0.35
427393009,0.3,0.3,0.325,0.5,0.325,0.4,0.5,0.5,0.45,0.45,0.5,0.375,0.35,0.45,0.45,0.4625,0.425,0.375,0.425,0.2,0.45,1.0,0.5,0.425,0.3,0.3
426177001,0.3,0.3,0.325,0.5,0.325,0.4,0.5,0.5,0.45,0.45,0.5,0.375,0.35,0.45,0.45,0.4625,0.425,0.375,0.425,0.2,0.45,0.5,1.0,0.425,0.3,0.3
427084000,0.375,0.375,0.4,0.425,0.4,0.475,0.425,0.425,0.475,0.475,0.425,0.45,0.425,0.475,0.375,0.4625,0.5,0.45,0.5,0.275,0.475,0.425,0.425,1.0,0.375,0.375
164934002,0.5,0.5,0.475,0.3,0.475,0.4,0.3,0.3,0.35,0.35,0.3,0.425,0.45,0.35,0.25,0.3375,0.375,0.425,0.375,0.4,0.35,0.3,0.3,0.375,1.0,0.5
59931005,0.5,0.5,0.475,0.3,0.475,0.4,0.3,0.3,0.35,0.35,0.3,0.425,0.45,0.35,0.25,0.3375,0.375,0.425,0.375,0.4,0.35,0.3,0.3,0.375,0.5,1.0"""
    ),
    index_col=0,
)
df_weights.index = df_weights.index.map(str)

def expand_equiv_classes(df: pd.DataFrame, sep: str = "|") -> pd.DataFrame:
    """finished, checked,
    expand df so that rows/cols with equivalent classes indicated by `sep` are separated
    Parameters
    ----------
    df: DataFrame,
        the dataframe to be split
    sep: str, default "|",
        separator of equivalent classes
    Returns
    -------
    df_out: DataFrame,
        the expanded DataFrame
    """
    # check whether df is symmetric
    if not (df.columns == df.index).all() or not (df.values.T == df.values).all():
        raise ValueError("the input DataFrame (matrix) is not symmetric")
    df_out = df.copy()
    col_row = df_out.columns.tolist()
    # df_sep = "\|" if sep == "|" else sep
    new_cols = []
    for c in col_row:
        for new_c in c.split(sep)[1:]:
            new_cols.append(new_c)
            df_out[new_c] = df_out[c].values
            new_r = new_c
            df_out.loc[new_r] = df_out.loc[df_out.index.str.contains(new_r)].values[0]
    col_row = [c.split(sep)[0] for c in col_row] + new_cols
    df_out.columns = col_row
    df_out.index = col_row
    return df_out

df_weights_expanded = expand_equiv_classes(df_weights)

dx_mapping_scored = pd.read_csv(
    StringIO(
        """Dx,SNOMEDCTCode,Abbreviation,CPSC,CPSC_Extra,StPetersburg,PTB,PTB_XL,Georgia,Chapman_Shaoxing,Ningbo,Total,Notes
atrial fibrillation,164889003,AF,1221,153,2,15,1514,570,1780,0,5255,
atrial flutter,164890007,AFL,0,54,0,1,73,186,445,7615,8374,
bundle branch block,6374002,BBB,0,0,1,20,0,116,0,385,522,
bradycardia,426627000,Brady,0,271,11,0,0,6,0,7,295,
complete left bundle branch block,733534002,CLBBB,0,0,0,0,0,0,0,213,213,We score 733534002 and 164909002 as the same diagnosis
complete right bundle branch block,713427006,CRBBB,0,113,0,0,542,28,0,1096,1779,We score 713427006 and 59118001 as the same diagnosis.
1st degree av block,270492004,IAVB,722,106,0,0,797,769,247,893,3534,
incomplete right bundle branch block,713426002,IRBBB,0,86,0,0,1118,407,0,246,1857,
left axis deviation,39732003,LAD,0,0,0,0,5146,940,382,1163,7631,
left anterior fascicular block,445118002,LAnFB,0,0,0,0,1626,180,0,380,2186,
left bundle branch block,164909002,LBBB,236,38,0,0,536,231,205,35,1281,We score 733534002 and 164909002 as the same diagnosis
low qrs voltages,251146004,LQRSV,0,0,0,0,182,374,249,794,1599,
nonspecific intraventricular conduction disorder,698252002,NSIVCB,0,4,1,0,789,203,235,536,1768,
sinus rhythm,426783006,NSR,918,4,0,80,18092,1752,1826,6299,28971,
premature atrial contraction,284470004,PAC,616,73,3,0,398,639,258,1054,3041,We score 284470004 and 63593006 as the same diagnosis.
pacing rhythm,10370003,PR,0,3,0,0,296,0,0,1182,1481,
poor R wave Progression,365413008,PRWP,0,0,0,0,0,0,0,638,638,
premature ventricular contractions,427172004,PVC,0,188,0,0,0,0,0,1091,1279,We score 427172004 and 17338001 as the same diagnosis.
prolonged pr interval,164947007,LPR,0,0,0,0,340,0,12,40,392,
prolonged qt interval,111975006,LQT,0,4,0,0,118,1391,57,337,1907,
qwave abnormal,164917005,QAb,0,1,0,0,548,464,235,828,2076,
right axis deviation,47665007,RAD,0,1,0,0,343,83,215,638,1280,
right bundle branch block,59118001,RBBB,1857,1,2,0,0,542,454,195,3051,We score 713427006 and 59118001 as the same diagnosis.
sinus arrhythmia,427393009,SA,0,11,2,0,772,455,0,2550,3790,
sinus bradycardia,426177001,SB,0,45,0,0,637,1677,3889,12670,18918,
sinus tachycardia,427084000,STach,0,303,11,1,826,1261,1568,5687,9657,
supraventricular premature beats,63593006,SVPB,0,53,4,0,157,1,0,9,224,We score 284470004 and 63593006 as the same diagnosis.
t wave abnormal,164934002,TAb,0,22,0,0,2345,2306,1876,5167,11716,
t wave inversion,59931005,TInv,0,5,1,0,294,812,157,2720,3989,
ventricular premature beats,17338001,VPB,0,8,0,0,0,357,294,0,659,We score 427172004 and 17338001 as the same diagnosis."""
    )
)
dx_mapping_scored = dx_mapping_scored.fillna("")
dx_mapping_scored["SNOMEDCTCode"] = dx_mapping_scored["SNOMEDCTCode"].apply(str)
dx_mapping_scored["CUSPHNFH"] = (
    dx_mapping_scored["Chapman_Shaoxing"].values + dx_mapping_scored["Ningbo"].values
)
dx_mapping_scored = dx_mapping_scored[
    "Dx,SNOMEDCTCode,Abbreviation,CPSC,CPSC_Extra,StPetersburg,PTB,PTB_XL,Georgia,CUSPHNFH,Chapman_Shaoxing,Ningbo,Total,Notes".split(
        ","
    )
]

dx_mapping_unscored = pd.read_csv(
    StringIO(
        """Dx,SNOMEDCTCode,Abbreviation,CPSC,CPSC_Extra,StPetersburg,PTB,PTB_XL,Georgia,Chapman_Shaoxing,Ningbo,Total
accelerated atrial escape rhythm,233892002,AAR,0,0,0,0,0,0,0,16,16
abnormal QRS,164951009,abQRS,0,0,0,0,3389,0,0,0,3389
atrial escape beat,251187003,AED,0,0,0,0,0,0,0,17,17
accelerated idioventricular rhythm,61277005,AIVR,0,0,0,0,0,0,0,14,14
accelerated junctional rhythm,426664006,AJR,0,0,0,0,0,19,0,12,31
suspect arm ecg leads reversed,251139008,ALR,0,0,0,0,0,12,0,0,12
acute myocardial infarction,57054005,AMI,0,0,6,0,0,0,0,49,55
acute myocardial ischemia,413444003,AMIs,0,1,0,0,0,1,0,0,2
anterior ischemia,426434006,AnMIs,0,0,0,0,44,281,0,0,325
anterior myocardial infarction,54329005,AnMI,0,62,0,0,354,0,0,57,473
atrial bigeminy,251173003,AB,0,0,3,0,0,0,3,0,6
atrial fibrillation and flutter,195080001,AFAFL,0,39,0,0,0,2,0,0,41
atrial hypertrophy,195126007,AH,0,2,0,0,0,60,0,0,62
atrial pacing pattern,251268003,AP,0,0,0,0,0,52,0,0,52
atrial rhythm,106068003,ARH,0,0,0,0,0,0,0,215,215
atrial tachycardia,713422000,ATach,0,15,0,0,0,28,121,176,340
av block,233917008,AVB,0,5,0,0,0,74,166,78,323
atrioventricular dissociation,50799005,AVD,0,0,0,0,0,0,0,59,59
atrioventricular junctional rhythm,29320008,AVJR,0,6,0,0,0,0,0,139,145
atrioventricular  node reentrant tachycardia,251166008,AVNRT,0,0,0,0,0,0,16,0,16
atrioventricular reentrant tachycardia,233897008,AVRT,0,0,0,0,0,0,8,18,26
blocked premature atrial contraction,251170000,BPAC,0,2,3,0,0,0,0,62,67
brugada,418818005,BRU,0,0,0,0,0,0,0,5,5
brady tachy syndrome,74615001,BTS,0,1,1,0,0,0,0,0,2
chronic atrial fibrillation,426749004,CAF,0,1,0,0,0,0,0,0,1
countercolockwise rotation,251199005,CCR,0,0,0,0,0,0,162,0,162
clockwise or counterclockwise vectorcardiographic loop,61721007,CVCL/CCVCL,0,0,0,0,0,0,0,653,653
cardiac dysrhythmia,698247007,CD,0,0,0,16,0,0,0,0,16
complete heart block,27885002,CHB,0,27,0,0,16,8,1,75,127
congenital incomplete atrioventricular heart block,204384007,CIAHB,0,0,0,2,0,0,0,0,2
coronary heart disease,53741008,CHD,0,0,16,21,0,0,0,0,37
chronic myocardial ischemia,413844008,CMI,0,161,0,0,0,0,0,0,161
clockwise rotation,251198002,CR,0,0,0,0,0,0,76,0,76
diffuse intraventricular block,82226007,DIB,0,1,0,0,0,0,0,0,1
early repolarization,428417006,ERe,0,0,0,0,0,140,22,344,506
fusion beats,13640000,FB,0,0,7,0,0,0,2,114,123
fqrs wave,164942001,FQRS,0,0,0,0,0,0,3,0,3
heart failure,84114007,HF,0,0,0,7,0,0,0,0,7
heart valve disorder,368009,HVD,0,0,0,6,0,0,0,0,6
high t-voltage,251259000,HTV,0,1,0,0,0,0,0,0,1
indeterminate cardiac axis,251200008,ICA,0,0,0,0,156,0,0,0,156
2nd degree av block,195042002,IIAVB,0,21,0,0,14,23,8,58,124
mobitz type II atrioventricular block,426183003,IIAVBII,0,0,0,0,0,0,0,7,7
inferior ischaemia,425419005,IIs,0,0,0,0,219,451,0,0,670
incomplete left bundle branch block,251120003,ILBBB,0,42,0,0,77,86,0,6,211
inferior ST segment depression,704997005,ISTD,0,1,0,0,0,0,0,0,1
idioventricular rhythm,49260003,IR,0,0,2,0,0,0,0,0,2
junctional escape,426995002,JE,0,4,0,0,0,5,15,60,84
junctional premature complex,251164006,JPC,0,2,0,0,0,0,1,10,13
junctional tachycardia,426648003,JTach,0,2,0,0,0,4,0,24,30
left atrial abnormality,253352002,LAA,0,0,0,0,0,72,0,0,72
left atrial enlargement,67741000119109,LAE,0,1,0,0,427,870,0,1,1299
left atrial hypertrophy,446813000,LAH,0,40,0,0,0,0,0,8,48
lateral ischaemia,425623009,LIs,0,0,0,0,142,903,0,0,1045
left posterior fascicular block,445211001,LPFB,0,0,0,0,177,25,0,5,207
left ventricular hypertrophy,164873001,LVH,0,158,10,0,2359,1232,15,632,4406
left ventricular high voltage,55827005,LVHV,0,0,0,0,0,0,1295,4106,5401
left ventricular strain,370365005,LVS,0,1,0,0,0,0,0,0,1
myocardial infarction,164865005,MI,0,376,9,368,5261,7,40,83,6144
myocardial ischemia,164861001,MIs,0,384,0,0,2175,0,0,0,2559
mobitz type i wenckebach atrioventricular block,54016002,MoI,0,0,3,0,0,0,6,25,34
nonspecific st t abnormality,428750005,NSSTTA,0,1290,0,0,381,1883,1158,0,4712
old myocardial infarction,164867002,OldMI,0,1168,0,0,0,0,0,0,1168
paroxysmal atrial fibrillation,282825002,PAF,0,0,1,1,0,0,0,0,2
prolonged P wave,251205003,PPW,0,0,0,0,0,0,0,106,106
paroxysmal supraventricular tachycardia,67198005,PSVT,0,0,3,0,24,0,0,0,27
paroxysmal ventricular tachycardia,425856008,PVT,0,0,15,0,0,0,0,109,124
p wave change,164912004,PWC,0,0,0,0,0,0,95,47,142
right atrial abnormality,253339007,RAAb,0,0,0,0,0,14,0,0,14
r wave abnormal,164921003,RAb,0,1,0,0,0,10,0,0,11
right atrial hypertrophy,446358003,RAH,0,18,0,0,99,0,3,33,153
right atrial  high voltage,67751000119106,RAHV,0,0,0,0,0,0,8,28,36
rapid atrial fibrillation,314208002,RAF,0,0,0,2,0,0,0,0,2
right ventricular hypertrophy,89792004,RVH,0,20,0,0,126,86,4,106,342
sinus atrium to atrial wandering rhythm,17366009,SAAWR,0,0,0,0,0,0,7,0,7
sinoatrial block,65778007,SAB,0,9,0,0,0,0,0,5,14
sinus arrest,5609005,SARR,0,0,0,0,0,0,0,33,33
sinus node dysfunction,60423000,SND,0,0,2,0,0,0,0,0,2
shortened pr interval,49578007,SPRI,0,3,0,0,0,2,0,23,28
decreased qt interval,77867006,SQT,0,1,0,0,0,0,0,2,3
s t changes,55930002,STC,0,1,0,0,770,6,0,4232,5009
st depression,429622005,STD,869,57,4,0,1009,38,402,1266,3645
st elevation,164931005,STE,220,66,4,0,28,134,176,0,628
st interval abnormal,164930006,STIAb,0,481,2,0,0,992,2,799,2276
supraventricular bigeminy,251168009,SVB,0,0,1,0,0,0,0,0,1
supraventricular tachycardia,426761007,SVT,0,3,1,0,27,32,587,137,787
transient ischemic attack,266257000,TIA,0,0,7,0,0,0,0,0,7
tall p wave,251223006,TPW,0,0,0,0,0,0,0,215,215
u wave abnormal,164937009,UAb,0,1,0,0,0,0,22,114,137
ventricular bigeminy,11157007,VBig,0,5,9,0,82,2,3,0,101
ventricular ectopics,164884008,VEB,700,0,49,0,1154,41,0,0,1944
ventricular escape beat,75532003,VEsB,0,3,1,0,0,0,7,49,60
ventricular escape rhythm,81898007,VEsR,0,1,0,0,0,1,0,96,98
ventricular fibrillation,164896001,VF,0,10,0,25,0,3,0,59,97
ventricular flutter,111288001,VFL,0,1,0,0,0,0,0,7,8
ventricular hypertrophy,266249003,VH,0,5,0,13,30,71,0,0,119
ventricular pre excitation,195060002,VPEx,0,6,0,0,0,2,12,0,20
ventricular pacing pattern,251266004,VPP,0,0,0,0,0,46,0,0,46
paired ventricular premature complexes,251182009,VPVC,0,0,23,0,0,0,0,0,23
ventricular tachycardia,164895002,VTach,0,1,1,10,0,0,0,0,12
ventricular trigeminy,251180001,VTrig,0,4,4,0,20,1,8,0,37
wandering atrial pacemaker,195101003,WAP,0,0,0,0,0,7,2,0,9
wolff parkinson white pattern,74390002,WPW,0,0,4,2,80,2,4,68,160"""
    )
)
dx_mapping_unscored["SNOMEDCTCode"] = dx_mapping_unscored["SNOMEDCTCode"].apply(str)
dx_mapping_unscored["CUSPHNFH"] = (
    dx_mapping_unscored["Chapman_Shaoxing"].values
    + dx_mapping_unscored["Ningbo"].values
)
dx_mapping_unscored = dx_mapping_unscored[
    "Dx,SNOMEDCTCode,Abbreviation,CPSC,CPSC_Extra,StPetersburg,PTB,PTB_XL,Georgia,CUSPHNFH,Chapman_Shaoxing,Ningbo,Total".split(
        ","
    )
]

dms = dx_mapping_scored.copy()
dms["scored"] = True
dmn = dx_mapping_unscored.copy()
dmn["Notes"] = ""
dmn["scored"] = False
dx_mapping_all = pd.concat([dms, dmn], ignore_index=True).fillna("")

snomed_ct_code_to_abbr = ED(
    {row["SNOMEDCTCode"]: row["Abbreviation"] for _, row in dx_mapping_all.iterrows()}
)
abbr_to_snomed_ct_code = ED({v: k for k, v in snomed_ct_code_to_abbr.items()})

df_weights_abbr = df_weights_expanded.copy()

df_weights_abbr.columns = df_weights_abbr.columns.map(
    lambda i: snomed_ct_code_to_abbr.get(i, i)
)
# df_weights_abbr.columns.map(lambda i: snomed_ct_code_to_abbr[i])

df_weights_abbr.index = df_weights_abbr.index.map(
    lambda i: snomed_ct_code_to_abbr.get(i, i)
)
# df_weights_abbr.index.map(lambda i: snomed_ct_code_to_abbr[i])

df_weights_abbreviations = (
    df_weights.copy()
)  # corresponding to weights_abbreviations.csv
df_weights_abbreviations.columns = df_weights_abbreviations.columns.map(
    lambda i: "|".join(
        [snomed_ct_code_to_abbr.get(item, item) for item in i.split("|")]
    )
)
# df_weights_abbreviations.columns.map(lambda i: "|".join([snomed_ct_code_to_abbr[item] for item in i.split("|")]))
df_weights_abbreviations.index = df_weights_abbreviations.index.map(
    lambda i: "|".join(
        [snomed_ct_code_to_abbr.get(item, item) for item in i.split("|")]
    )
)
# df_weights_abbreviations.index.map(lambda i: "|".join([snomed_ct_code_to_abbr[item] for item in i.split("|")]))


snomed_ct_code_to_fullname = ED(
    {row["SNOMEDCTCode"]: row["Dx"] for _, row in dx_mapping_all.iterrows()}
)
fullname_to_snomed_ct_code = ED({v: k for k, v in snomed_ct_code_to_fullname.items()})

df_weights_fullname = df_weights_expanded.copy()

df_weights_fullname.columns = df_weights_fullname.columns.map(
    lambda i: snomed_ct_code_to_fullname.get(i, i)
)
# df_weights_fullname.columns.map(lambda i: snomed_ct_code_to_fullname[i])

df_weights_fullname.index = df_weights_fullname.index.map(
    lambda i: snomed_ct_code_to_fullname.get(i, i)
)
# df_weights_fullname.index.map(lambda i: snomed_ct_code_to_fullname[i])


abbr_to_fullname = ED(
    {row["Abbreviation"]: row["Dx"] for _, row in dx_mapping_all.iterrows()}
)
fullname_to_abbr = ED({v: k for k, v in abbr_to_fullname.items()})


equiv_class_dict = {}
for c in df_weights.columns:
    if "|" not in c:
        continue
    v, k = c.split("|")
    equiv_class_dict[k] = v
    equiv_class_dict[snomed_ct_code_to_abbr[k]] = snomed_ct_code_to_abbr[v]
    equiv_class_dict[snomed_ct_code_to_fullname[k]] = snomed_ct_code_to_fullname[v]


def get_class_weight(
    tranches: Union[str, Sequence[str]],
    exclude_classes: Optional[Sequence[str]] = None,
    scored_only: bool = False,
    normalize: bool = True,
    threshold: Optional[Real] = 0,
    fmt: str = "a",
    min_weight: Real = 0.5,
) -> Dict[str, int]:
    """finished, checked,
    Parameters
    ----------
    tranches: str or sequence of str,
        tranches to count classes, can be combinations of "A", "B", "C", "D", "E", "F"
    exclude_classes: sequence of str, optional,
        abbrevations or SNOMEDCTCodes of classes to be excluded from counting
    scored_only: bool, default True,
        if True, only scored classes are counted
    normalize: bool, default True,
        collapse equivalent classes into one,
        used only when `scored_only` = True
    threshold: real number,
        minimum ratio (0-1) or absolute number (>1) of a class to be counted
    fmt: str, default "a",
        the format of the names of the classes in the returned dict,
        can be one of the following (case insensitive):
        - "a", abbreviations
        - "f", full names
        - "s", SNOMEDCTCode
    min_weight: real number, default 0.5,
        minimum value of the weight of all classes,
        or equivalently the weight of the largest class
    Returns
    -------
    class_weight: dict,
        key: class in the format of `fmt`
        value: weight of a class in `tranches`
    """
    class_count = get_class_count(
        tranches=tranches,
        exclude_classes=exclude_classes,
        scored_only=scored_only,
        normalize=normalize,
        threshold=threshold,
        fmt=fmt,
    )
    class_weight = ED(
        {key: sum(class_count.values()) / val for key, val in class_count.items()}
    )
    class_weight = ED(
        {
            key: min_weight * val / min(class_weight.values())
            for key, val in class_weight.items()
        }
    )
    return class_weight


def get_class_count(
    tranches: Union[str, Sequence[str]],
    exclude_classes: Optional[Sequence[str]] = None,
    scored_only: bool = False,
    normalize: bool = True,
    threshold: Optional[Real] = 0,
    fmt: str = "a",
) -> Dict[str, int]:
    """finished, checked,
    Parameters
    ----------
    tranches: str or sequence of str,
        tranches to count classes, can be combinations of "A", "B", "C", "D", "E", "F", "G"
    exclude_classes: sequence of str, optional,
        abbrevations or SNOMEDCTCodes of classes to be excluded from counting
    scored_only: bool, default True,
        if True, only scored classes are counted
    normalize: bool, default True,
        collapse equivalent classes into one,
        used only when `scored_only` = True
    threshold: real number,
        minimum ratio (0-1) or absolute number (>1) of a class to be counted
    fmt: str, default "a",
        the format of the names of the classes in the returned dict,
        can be one of the following (case insensitive):
        - "a", abbreviations
        - "f", full names
        - "s", SNOMEDCTCode
    Returns
    -------
    class_count: dict,
        key: class in the format of `fmt`
        value: count of a class in `tranches`
    """
    assert threshold >= 0
    tranche_names = ED(
        {
            "A": "CPSC",
            "B": "CPSC_Extra",
            "C": "StPetersburg",
            "D": "PTB",
            "E": "PTB_XL",
            "F": "Georgia",
            "G": "CUSPHNFH",
        }
    )
    tranche_names = [tranche_names[t] for t in tranches]
    _exclude_classes = [normalize_class(c) for c in (exclude_classes or [])]
    df = dx_mapping_scored.copy() if scored_only else dx_mapping_all.copy()
    class_count = ED()
    for _, row in df.iterrows():
        key = row["Abbreviation"]
        val = row[tranche_names].values.sum()
        if val == 0:
            continue
        if key in _exclude_classes:
            continue
        if normalize and scored_only:
            key = equiv_class_dict.get(key, key)
        if key in _exclude_classes:
            continue
        if key in class_count.keys():
            class_count[key] += val
        else:
            class_count[key] = val
    tmp = ED()
    tot_count = sum(class_count.values())
    _threshold = threshold if threshold >= 1 else threshold * tot_count
    if fmt.lower() == "s":
        for key, val in class_count.items():
            if val < _threshold:
                continue
            tmp[abbr_to_snomed_ct_code[key]] = val
        class_count = tmp.copy()
    elif fmt.lower() == "f":
        for key, val in class_count.items():
            if val < _threshold:
                continue
            tmp[abbr_to_fullname[key]] = val
        class_count = tmp.copy()
    else:
        class_count = {
            key: val for key, val in class_count.items() if val >= _threshold
        }
    del tmp
    return class_count


def normalize_class(c: Union[str, int], ensure_scored: bool = False) -> str:
    """finished, checked,
    normalize the class name to its abbr.,
    facilitating the computation of the `load_weights` function
    Parameters
    ----------
    c: str or int,
        abbr. or SNOMEDCTCode of the class
    ensure_scored: bool, default False,
        ensure that the class is a scored class,
        if True, `ValueError` would be raised if `c` is not scored
    Returns
    -------
    nc: str,
        the abbr. of the class
    """
    nc = snomed_ct_code_to_abbr.get(str(c), str(c))
    if ensure_scored and nc not in df_weights_abbr.columns:
        raise ValueError(f"class `{c}` not among the scored classes")
    return nc

def ensure_siglen(
    values: Sequence[Real],
    siglen: int,
    fmt: str = "lead_first",
    tolerance: Optional[float] = None,
) -> np.ndarray:
    """finished, checked,
    ensure the (ECG) signal to be of length `siglen`,
    strategy:
        If `values` has length greater than `siglen`,
        the central `siglen` samples will be adopted;
        otherwise, zero padding will be added to both sides.
        If `tolerance` is given,
        then if the length of `values` is longer than `siglen` by more than `tolerance`,
        the `values` will be sliced to have multiple of `siglen` samples.
    Parameters
    ----------
    values: sequence,
        values of the `n_leads`-lead (ECG) signal
    siglen: int,
        length of the signal supposed to have
    fmt: str, default "lead_first", case insensitive,
        format of the input and output values, can be one of
        "lead_first" (alias "channel_first"), "lead_last" (alias "channel_last")
    Returns
    -------
    out_values: ndarray,
        ECG signal in the format of `fmt` and of fixed length `siglen`,
        of ndim=3 if `tolerence` is given, otherwise ndim=2
    """
    if fmt.lower() in ["channel_last", "lead_last"]:
        _values = np.array(values).T
    else:
        _values = np.array(values).copy()
    original_siglen = _values.shape[1]
    n_leads = _values.shape[0]

    if tolerance is None or original_siglen <= siglen * (1 + tolerance):
        if original_siglen >= siglen:
            start = (original_siglen - siglen) // 2
            end = start + siglen
            out_values = _values[..., start:end]
        else:
            pad_len = siglen - original_siglen
            pad_left = pad_len // 2
            pad_right = pad_len - pad_left
            out_values = np.concatenate(
                [
                    np.zeros((n_leads, pad_left)),
                    _values,
                    np.zeros((n_leads, pad_right)),
                ],
                axis=1,
            )

        if fmt.lower() in ["channel_last", "lead_last"]:
            out_values = out_values.T
        if tolerance is not None:
            out_values = out_values[np.newaxis, ...]

        return out_values

    forward_len = int(round(siglen * tolerance))
    out_values = np.array(
        [
            _values[..., idx * forward_len : idx * forward_len + siglen]
            for idx in range((original_siglen - siglen) // forward_len + 1)
        ]
    )
    if fmt.lower() in ["channel_last", "lead_last"]:
        out_values = np.moveaxis(out_values, 1, -1)
    return out_values

def list_sum(lst: Sequence[list]) -> list:
    """finished, checked,
    Parameters
    ----------
    lst: sequence of list,
        the sequence of lists to obtain the summation
    Returns
    -------
    l_sum: list,
        sum of `lst`,
        i.e. if lst = [list1, list2, ...], then l_sum = list1 + list2 + ...
    """
    l_sum = reduce(lambda a, b: a + b, lst, [])
    return l_sum

def remove_spikes_naive(sig: np.ndarray) -> np.ndarray:
    """finished, checked,
    remove `spikes` from `sig` using a naive method proposed in entry 0416 of CPSC2019
    `spikes` here refers to abrupt large bumps with (abs) value larger than 20 mV,
    or nan values (read by `wfdb`),
    do NOT confuse with `spikes` in paced rhythm
    Parameters
    ----------
    sig: ndarray,
        single-lead ECG signal with potential spikes
    Returns
    -------
    filtered_sig: ndarray,
        ECG signal with `spikes` removed
    """
    b = list(
        filter(
            lambda k: k > 0,
            np.argwhere(np.logical_or(np.abs(sig) > 20, np.isnan(sig))).squeeze(-1),
        )
    )
    filtered_sig = sig.copy()
    for k in b:
        filtered_sig[k] = filtered_sig[k - 1]
    return filtered_sig

class ReprMixin(object):
    """
    Mixin for enhanced __repr__ and __str__ methods.
    """

    def __repr__(self) -> str:
        return default_class_repr(self)

    __str__ = __repr__

    def extra_repr_keys(self) -> List[str]:
        """ """
        return []

def load_weights(
    classes: Sequence[Union[int, str]] = None,
    equivalent_classes: Optional[Union[Dict[str, str], List[List[str]]]] = None,
    return_fmt: str = "np",
) -> Union[np.ndarray, pd.DataFrame]:
    """NOT finished, NOT checked,
    load the weight matrix of the `classes`
    Parameters
    ----------
    classes: sequence of str or int, optional,
        the classes (abbr. or SNOMEDCTCode) to load their weights,
        if not given, weights of all classes in `dx_mapping_scored` will be loaded
    equivalent_classes: dict or list, optional,
        list or dict of equivalent classes,
        if not specified, defaults to `equiv_class_dict`
    return_fmt: str, default "np",
        "np" or "pd", the values in the form of a 2d array or a DataFrame
    Returns
    -------
    mat: 2d array or DataFrame,
        the weight matrix of the `classes`
    """
    if classes:
        l_nc = [normalize_class(c, ensure_scored=True) for c in classes]
        assert len(set(l_nc)) == len(classes), "`classes` has duplicates!"
        mat = df_weights_abbr.loc[l_nc, l_nc]
    else:
        mat = df_weights_abbr.copy()

    if return_fmt.lower() == "np":
        mat = mat.values
    elif return_fmt.lower() == "pd":
        # columns and indices back to the original input format
        mat.columns = list(map(str, classes))
        mat.index = list(map(str, classes))
    else:
        raise ValueError(f"format of `{return_fmt}` is not supported!")

    return mat


def evaluate_scores_detailed(
    truth: Sequence, 
    # binary_pred: Sequence, 
    scalar_pred: Sequence
) -> Tuple[Union[float, np.ndarray]]:
    """finished, checked,
    Parameters
    ----------
    classes: list of str,
        list of all the classes, in the format of abbrevations
    truth: sequence,
        ground truth array, of shape (n_records, n_classes), with values 0 or 1
    binary_pred: sequence,
        binary predictions, of shape (n_records, n_classes), with values 0 or 1
    scalar_pred: sequence,
        probability predictions, of shape (n_records, n_classes), with values within [0,1]
    Returns
    -------
    auroc: float,
    auprc: float,
    auroc_classes: ndarray,
    auprc_classes: ndarray,
    accuracy: float,
    f_measure: float,
    f_measure_classes: ndarray,
    f_beta_measure: float,
    g_beta_measure: float,
    challenge_metric: float,
    """

    classes = df_weights_abbr.columns.tolist()
    # sinus_rhythm = "426783006"
    sinus_rhythm = "NSR"
    weights = load_weights(classes=classes)

    _truth = np.array(truth)
    _binary_pred = (np.array(scalar_pred) > 0.5).astype(float)
    _scalar_pred = np.array(scalar_pred)

    auroc, auprc, auroc_classes, auprc_classes = compute_auc(_truth, _scalar_pred)

    accuracy = compute_accuracy(_truth, _binary_pred)

    f_measure, f_measure_classes = compute_f_measure(_truth, _binary_pred)

    f_beta_measure, g_beta_measure = compute_beta_measures(_truth, _binary_pred, beta=2)

    challenge_metric = 0 #compute_challenge_metric(weights, _truth, _binary_pred, classes, sinus_rhythm)

    # Return the results.
    ret_tuple = (
        auroc,
        auprc,
        auroc_classes,
        auprc_classes,
        accuracy,
        f_measure,
        f_measure_classes,
        f_beta_measure,
        g_beta_measure,
        challenge_metric,
    )
    return ret_tuple


def evaluate_scores(
    truth: Sequence, 
    # binary_pred: Sequence, 
    scalar_pred: Sequence
) -> Tuple[Union[float, np.ndarray]]:
    """finished, checked,
    simplified version of `evaluate_scores_detailed`,
    this function doesnot produce per class scores
    Parameters
    ----------
    classes: list of str,
        list of all the classes, in the format of abbrevations
    truth: sequence,
        ground truth array, of shape (n_records, n_classes), with values 0 or 1
    binary_pred: sequence,
        binary predictions, of shape (n_records, n_classes), with values 0 or 1
    scalar_pred: sequence,
        probability predictions, of shape (n_records, n_classes), with values within [0,1]
    Returns
    -------
    auroc: float,
    auprc: float,
    accuracy: float,
    f_measure: float,
    f_beta_measure: float,
    g_beta_measure: float,
    challenge_metric: float,
    """
    
    (
        auroc,
        auprc,
        _,
        _,
        accuracy,
        f_measure,
        _,
        f_beta_measure,
        g_beta_measure,
        challenge_metric,
    ) = evaluate_scores_detailed(truth, scalar_pred)
    return (
        auroc,
        # auprc,
        accuracy,
        # f_measure,
        # f_beta_measure,
        # g_beta_measure,
        # challenge_metric,
    )


# Compute recording-wise accuracy.
def compute_accuracy(labels: np.ndarray, outputs: np.ndarray) -> float:
    """checked,"""
    num_recordings, num_classes = np.shape(labels)

    num_correct_recordings = 0
    for i in range(num_recordings):
        if np.all(labels[i, :] == outputs[i, :]):
            num_correct_recordings += 1

    return float(num_correct_recordings) / float(num_recordings)


# Compute confusion matrices.
def compute_confusion_matrices(
    labels: np.ndarray, outputs: np.ndarray, normalize: bool = False
) -> np.ndarray:
    """checked,"""
    # Compute a binary confusion matrix for each class k:
    #
    #     [TN_k FN_k]
    #     [FP_k TP_k]
    #
    # If the normalize variable is set to true, then normalize the contributions
    # to the confusion matrix by the number of labels per recording.
    num_recordings, num_classes = np.shape(labels)

    if not normalize:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            for j in range(num_classes):
                if labels[i, j] == 1 and outputs[i, j] == 1:  # TP
                    A[j, 1, 1] += 1
                elif labels[i, j] == 0 and outputs[i, j] == 1:  # FP
                    A[j, 1, 0] += 1
                elif labels[i, j] == 1 and outputs[i, j] == 0:  # FN
                    A[j, 0, 1] += 1
                elif labels[i, j] == 0 and outputs[i, j] == 0:  # TN
                    A[j, 0, 0] += 1
                else:  # This condition should not happen.
                    raise ValueError("Error in computing the confusion matrix.")
    else:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            normalization = float(max(np.sum(labels[i, :]), 1))
            for j in range(num_classes):
                if labels[i, j] == 1 and outputs[i, j] == 1:  # TP
                    A[j, 1, 1] += 1.0 / normalization
                elif labels[i, j] == 0 and outputs[i, j] == 1:  # FP
                    A[j, 1, 0] += 1.0 / normalization
                elif labels[i, j] == 1 and outputs[i, j] == 0:  # FN
                    A[j, 0, 1] += 1.0 / normalization
                elif labels[i, j] == 0 and outputs[i, j] == 0:  # TN
                    A[j, 0, 0] += 1.0 / normalization
                else:  # This condition should not happen.
                    raise ValueError("Error in computing the confusion matrix.")

    return A


# Compute macro F-measure.
def compute_f_measure(
    labels: np.ndarray, outputs: np.ndarray
) -> Tuple[float, np.ndarray]:
    """checked,"""
    num_recordings, num_classes = np.shape(labels)

    A = compute_confusion_matrices(labels, outputs)

    f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
        if 2 * tp + fp + fn:
            f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            f_measure[k] = float("nan")

    if np.any(np.isfinite(f_measure)):
        macro_f_measure = np.nanmean(f_measure)
    else:
        macro_f_measure = float("nan")

    return macro_f_measure, f_measure


# Compute F-beta and G-beta measures from the unofficial phase of the Challenge.
def compute_beta_measures(
    labels: np.ndarray, outputs: np.ndarray, beta: Real
) -> Tuple[float, float]:
    """checked,"""
    num_recordings, num_classes = np.shape(labels)

    A = compute_confusion_matrices(labels, outputs, normalize=True)

    f_beta_measure = np.zeros(num_classes)
    g_beta_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
        if (1 + beta**2) * tp + fp + beta**2 * fn:
            f_beta_measure[k] = float((1 + beta**2) * tp) / float(
                (1 + beta**2) * tp + fp + beta**2 * fn
            )
        else:
            f_beta_measure[k] = float("nan")
        if tp + fp + beta * fn:
            g_beta_measure[k] = float(tp) / float(tp + fp + beta * fn)
        else:
            g_beta_measure[k] = float("nan")

    macro_f_beta_measure = np.nanmean(f_beta_measure)
    macro_g_beta_measure = np.nanmean(g_beta_measure)

    return macro_f_beta_measure, macro_g_beta_measure


# Compute macro AUROC and macro AUPRC.
def compute_auc(
    labels: np.ndarray, outputs: np.ndarray
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """checked,"""
    num_recordings, num_classes = np.shape(labels)

    # Compute and summarize the confusion matrices for each class across at distinct output values.
    auroc = np.zeros(num_classes)
    auprc = np.zeros(num_classes)

    for k in range(num_classes):
        # We only need to compute TPs, FPs, FNs, and TNs at distinct output values.
        thresholds = np.unique(outputs[:, k])
        thresholds = np.append(thresholds, thresholds[-1] + 1)
        thresholds = thresholds[::-1]
        num_thresholds = len(thresholds)

        # Initialize the TPs, FPs, FNs, and TNs.
        tp = np.zeros(num_thresholds)
        fp = np.zeros(num_thresholds)
        fn = np.zeros(num_thresholds)
        tn = np.zeros(num_thresholds)
        fn[0] = np.sum(labels[:, k] == 1)
        tn[0] = np.sum(labels[:, k] == 0)

        # Find the indices that result in sorted output values.
        idx = np.argsort(outputs[:, k])[::-1]

        # Compute the TPs, FPs, FNs, and TNs for class k across thresholds.
        i = 0
        for j in range(1, num_thresholds):
            # Initialize TPs, FPs, FNs, and TNs using values at previous threshold.
            tp[j] = tp[j - 1]
            fp[j] = fp[j - 1]
            fn[j] = fn[j - 1]
            tn[j] = tn[j - 1]

            # Update the TPs, FPs, FNs, and TNs at i-th output value.
            while i < num_recordings and outputs[idx[i], k] >= thresholds[j]:
                if labels[idx[i], k]:
                    tp[j] += 1
                    fn[j] -= 1
                else:
                    fp[j] += 1
                    tn[j] -= 1
                i += 1

        # Summarize the TPs, FPs, FNs, and TNs for class k.
        tpr = np.zeros(num_thresholds)
        tnr = np.zeros(num_thresholds)
        ppv = np.zeros(num_thresholds)
        for j in range(num_thresholds):
            if tp[j] + fn[j]:
                tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
            else:
                tpr[j] = float("nan")
            if fp[j] + tn[j]:
                tnr[j] = float(tn[j]) / float(fp[j] + tn[j])
            else:
                tnr[j] = float("nan")
            if tp[j] + fp[j]:
                ppv[j] = float(tp[j]) / float(tp[j] + fp[j])
            else:
                ppv[j] = float("nan")

        # Compute AUROC as the area under a piecewise linear function with TPR/
        # sensitivity (x-axis) and TNR/specificity (y-axis) and AUPRC as the area
        # under a piecewise constant with TPR/recall (x-axis) and PPV/precision
        # (y-axis) for class k.
        for j in range(num_thresholds - 1):
            auroc[k] += 0.5 * (tpr[j + 1] - tpr[j]) * (tnr[j + 1] + tnr[j])
            auprc[k] += (tpr[j + 1] - tpr[j]) * ppv[j + 1]

    # Compute macro AUROC and macro AUPRC across classes.
    if np.any(np.isfinite(auroc)):
        macro_auroc = np.nanmean(auroc)
    else:
        macro_auroc = float("nan")
    if np.any(np.isfinite(auprc)):
        macro_auprc = np.nanmean(auprc)
    else:
        macro_auprc = float("nan")

    return macro_auroc, macro_auprc, auroc, auprc


# Compute modified confusion matrix for multi-class, multi-label tasks.
def compute_modified_confusion_matrix(
    labels: np.ndarray, outputs: np.ndarray
) -> np.ndarray:
    """checked,
    Compute a binary multi-class, multi-label confusion matrix,
    where the rows are the labels and the columns are the outputs.
    """
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        normalization = float(
            max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1)
        )
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0 / normalization

    return A


# Compute the evaluation metric for the Challenge.
def compute_challenge_metric(
    weights: np.ndarray,
    labels: np.ndarray,
    outputs: np.ndarray,
    classes: List[str],
    sinus_rhythm: str,
) -> float:
    """checked,"""
    num_recordings, num_classes = np.shape(labels)
    if sinus_rhythm in classes:
        sinus_rhythm_index = classes.index(sinus_rhythm)
    else:
        raise ValueError("The sinus rhythm class is not available.")

    # Compute the observed score.
    A = compute_modified_confusion_matrix(labels, outputs)
    observed_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the correct label(s).
    correct_outputs = labels
    A = compute_modified_confusion_matrix(labels, correct_outputs)
    correct_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the sinus rhythm class.
    inactive_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
    inactive_outputs[:, sinus_rhythm_index] = 1
    A = compute_modified_confusion_matrix(labels, inactive_outputs)
    inactive_score = np.nansum(weights * A)

    if correct_score != inactive_score:
        normalized_score = float(observed_score - inactive_score) / float(
            correct_score - inactive_score
        )
    else:
        normalized_score = 0.0

    return normalized_score