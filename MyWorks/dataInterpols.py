import pandas
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import interpolate

dataSS = pandas.read_csv('/home/dynamo/a/debnathm/func_modelling/puma_functional_model/Simdata/SS_ADC_10u.csv')
# dataSS = pandas.read_csv('/home/dynamo/a/debnathm/func_modelling/puma_functional_model/Simdata/SS_ADC_all_corners_after_calibration.csv')

dataSS['delay_switch'] = dataSS['delay_switch'] - dataSS['delay_switch'].min()
adc_in_max_SS = dataSS['Iinput'].max()
adc_in_min_SS = dataSS['Iinput'].min()
tSS = dataSS['delay_switch'].max() - dataSS['delay_switch'].min()
DataNom = dataSS[dataSS['Corner']=='nom']
DataNomIn = DataNom['Iinput']
DataNomOut = DataNom['delay_switch']
InterpolFSS = interpolate.interp1d(DataNomIn, DataNomOut)

# dataSS['nom'] = dataSS['nom'] - 3.55e-9
# dataSS['C0'] = dataSS['C0'] - 3.55e-9
# dataSS['C1'] = dataSS['C1'] - 3.55e-9
# dataSS['C2'] = dataSS['C2'] - 3.55e-9
# dataSS['C3'] = dataSS['C3'] - 3.55e-9

# adc_in_range_SS = 1e-5
# tSS = 4.655e-8 - 3.55e-9
# DataNomIn = dataSS['Iinput']
# DataNomOut = dataSS['C2']
# InterpolFSS = interpolate.interp1d(DataNomIn, DataNomOut)

dataSSMC = pandas.read_csv('/home/dynamo/a/debnathm/func_modelling/puma_functional_model/Simdata/MC_run_for_range_0to10u_ExplorerRun.0.csv')
dataSSMC['delay_switch'] = dataSSMC['delay_switch'] - dataSSMC['delay_switch'].min()
tSSMC = dataSSMC['delay_switch'].max() - dataSSMC['delay_switch'].min()
mc_index = 97
dataMC = dataSSMC[dataSSMC['mc_iteration']==mc_index]
InterpolFSSMC = interpolate.interp1d(dataMC['Iinput'], dataMC['delay_switch'])

dataSSRamp = pandas.read_csv('/home/dynamo/a/debnathm/func_modelling/puma_functional_model/Simdata/Data_10thJuly.csv')
dataSSRamp['delay'] = dataSSRamp['delay'] - 60e-9
adc_in_max_SSRamp = dataSSRamp['Iinput'].max()
adc_in_min_SSRamp = dataSSRamp['Iinput'].min()
tSSRamp = 128e-9
InterpolFSSRamp = interpolate.interp1d(dataSSRamp['Iinput'], dataSSRamp['delay'])