# -*- coding: utf-8 -*-
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge, LinearRegression, BayesianRidge, RidgeClassifier
from sklearn.metrics import roc_auc_score, f1_score, plot_roc_curve
import xgboost as xgb
from sklearn.metrics import auc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(0)

cols = ["age_admission_unz", "triage", "referral_unz", "admission_choice",
"systolic_bp_first", "systolic_bp_first_inf_100", "diastolic_bp_first", 
"respiratory_rate_first", "gcs_inf_15_first", "lvl_consc_alert", 
"temperature_lowest", "spo2_first", "THZ (Thrombozyten)", "NA", "ASAT", 
"UREA = Harnstoff", "GFR", "PCThs", "EOS =eosinophile", "CRP", "KA = Kalium", 
"Hb", "INRiH", "CR", "pH", "LACT =lactat", "GGT", "BIC_st", "leuk_sup_10", 
"leuk_inf_4", "leuk_inf_0_5", "LACT =lactatinf7", "GGTinf5", "INRiHinf5", 
"THZ (Thrombozyten)inf3", "ASATinf3", "UREA = Harnstoffinf3", 
"ort_vor_aufnahme_spitalinf2", "BIC_stinf1",  "pHinf1", "CRinf8", 
"EOS =eosinophileinf6","diastolic_bp_firstinf4", "respiratory_rate_firstinf3", 
"systolic_bp_firstinf4", "NCH", "PLWS", "PSYCH", 'Hausarzt (privat)',
'Inselklinik', 'Polizei (In Begleitung)']

all_possible = ['age_admission_unz', 'sex', 'triage', 'admission_choice', 'entrance_unz', 'time_unz', 'entrance_insel', 'exit_insel', 'referral_unz', 'ort_vor_aufnahme_spital', 'pulse_first', 'pulse_highest', 'frequency_first', 'frequency_highest', 'systolic_bp_first', 'systolic_bp_lowest', 'systolic_bp_first_inf_100', 'systolic_bp_lowest_inf_100', 'diastolic_bp_first', 'diastolic_bp_lowest', 'respiratory_rate_first', 'respiratory_rate_highest', 'gcs_inf_15_first', 'gcs_inf_15_lowest', 'lvl_consc_alert', 'temperature_highest', 'temperature_lowest', 'temperature_sup_38', 'temperature_inf_35', 'spo2_first', 'spo2_lowest', 'o2_gabe', 'THZ (Thrombozyten)', 'NA', 'ASAT', 'UREA = Harnstoff', 'GFR', 'PCThs', 'EOS =eosinophile', 'CRP', 'KA = Kalium', 'Hb', 'ALAT', 'INRiH', 'CR', 'pH', 'LACT =lactat', 'GGT', 'BIC_st', 'BIC_VB', 'Leuk', 'leuk_sup_10', 'leuk_inf_4', 'leuk_inf_0_5', 'NRBCm', 'NRBCm_10_percent', 'aids', 'AUGEN', 'CHIR', 'FTN', 'HCH', 'HGC', 'HNO', 'KARD', 'KIND', 'MED', 'NCH', 'NEURO', 'ORTHO', 'PLWS', 'PSYCH', 'SKG', 'SRCHIR', 'SRMED', 'TCH', 'URO', 'VCH', 'Ambulanz', 'City-Notfall', 'Externes Spital (Ambulanz)', 'Externes Spital (privat)', 'Hausarzt (Ambulanz)', 'Hausarzt (privat)', 'Inselklinik', 'Polizei (In Begleitung)', 'Polizei (Strafanstalten)', 'Psychiatrie', 'Rega (Externes Spital)', 'Rega (Primär)', 'Rega (Repatriierung)', 'Selbsteinweisung', 'Sonstiges', 'UNZ Ambulatorium', 'age_admission_unzinf1', 'age_admission_unzinf2', 'age_admission_unzinf3', 'age_admission_unzinf4', 'age_admission_unzinf5', 'age_admission_unzinf6', 'age_admission_unzinf7', 'age_admission_unzinf8', 'age_admission_unzinf9', 'sexinf1', 'sexinf2', 'sexinf3', 'sexinf4', 'sexinf5', 'sexinf6', 'sexinf7', 'sexinf8', 'sexinf9', 'triageinf1', 'triageinf2', 'triageinf3', 'triageinf4', 'triageinf5', 'triageinf6', 'triageinf7', 'triageinf8', 'triageinf9', 'admission_choiceinf1', 'admission_choiceinf2', 'admission_choiceinf3', 'admission_choiceinf4', 'admission_choiceinf5', 'admission_choiceinf6', 'admission_choiceinf7', 'admission_choiceinf8', 'admission_choiceinf9', 'entrance_unzinf1', 'entrance_unzinf2', 'entrance_unzinf3', 'entrance_unzinf4', 'entrance_unzinf5', 'entrance_unzinf6', 'entrance_unzinf7', 'entrance_unzinf8', 'entrance_unzinf9', 'time_unzinf1', 'time_unzinf2', 'time_unzinf3', 'time_unzinf4', 'time_unzinf5', 'time_unzinf6', 'time_unzinf7', 'time_unzinf8', 'time_unzinf9', 'entrance_inselinf1', 'entrance_inselinf2', 'entrance_inselinf3', 'entrance_inselinf4', 'entrance_inselinf5', 'entrance_inselinf6', 'entrance_inselinf7', 'entrance_inselinf8', 'entrance_inselinf9', 'exit_inselinf1', 'exit_inselinf2', 'exit_inselinf3', 'exit_inselinf4', 'exit_inselinf5', 'exit_inselinf6', 'exit_inselinf7', 'exit_inselinf8', 'exit_inselinf9', 'referral_unzinf1', 'referral_unzinf2', 'referral_unzinf3', 'referral_unzinf4', 'referral_unzinf5', 'referral_unzinf6', 'referral_unzinf7', 'referral_unzinf8', 'referral_unzinf9', 'ort_vor_aufnahme_spitalinf1', 'ort_vor_aufnahme_spitalinf2', 'ort_vor_aufnahme_spitalinf3', 'ort_vor_aufnahme_spitalinf4', 'ort_vor_aufnahme_spitalinf5', 'ort_vor_aufnahme_spitalinf6', 'ort_vor_aufnahme_spitalinf7', 'ort_vor_aufnahme_spitalinf8', 'ort_vor_aufnahme_spitalinf9', 'hospitalisationinf1', 'hospitalisationinf2', 'hospitalisationinf3', 'hospitalisationinf4', 'hospitalisationinf5', 'hospitalisationinf6', 'hospitalisationinf7', 'hospitalisationinf8', 'hospitalisationinf9', 'ICUinf1', 'ICUinf2', 'ICUinf3', 'ICUinf4', 'ICUinf5', 'ICUinf6', 'ICUinf7', 'ICUinf8', 'ICUinf9', 'pulse_firstinf1', 'pulse_firstinf2', 'pulse_firstinf3', 'pulse_firstinf4', 'pulse_firstinf5', 'pulse_firstinf6', 'pulse_firstinf7', 'pulse_firstinf8', 'pulse_firstinf9', 'pulse_highestinf1', 'pulse_highestinf2', 'pulse_highestinf3', 'pulse_highestinf4', 'pulse_highestinf5', 'pulse_highestinf6', 'pulse_highestinf7', 'pulse_highestinf8', 'pulse_highestinf9', 'frequency_firstinf1', 'frequency_firstinf2', 'frequency_firstinf3', 'frequency_firstinf4', 'frequency_firstinf5', 'frequency_firstinf6', 'frequency_firstinf7', 'frequency_firstinf8', 'frequency_firstinf9', 'frequency_highestinf1', 'frequency_highestinf2', 'frequency_highestinf3', 'frequency_highestinf4', 'frequency_highestinf5', 'frequency_highestinf6', 'frequency_highestinf7', 'frequency_highestinf8', 'frequency_highestinf9', 'systolic_bp_firstinf1', 'systolic_bp_firstinf2', 'systolic_bp_firstinf3', 'systolic_bp_firstinf4', 'systolic_bp_firstinf5', 'systolic_bp_firstinf6', 'systolic_bp_firstinf7', 'systolic_bp_firstinf8', 'systolic_bp_firstinf9', 'systolic_bp_lowestinf1', 'systolic_bp_lowestinf2', 'systolic_bp_lowestinf3', 'systolic_bp_lowestinf4', 'systolic_bp_lowestinf5', 'systolic_bp_lowestinf6', 'systolic_bp_lowestinf7', 'systolic_bp_lowestinf8', 'systolic_bp_lowestinf9', 'systolic_bp_first_inf_100inf1', 'systolic_bp_first_inf_100inf2', 'systolic_bp_first_inf_100inf3', 'systolic_bp_first_inf_100inf4', 'systolic_bp_first_inf_100inf5', 'systolic_bp_first_inf_100inf6', 'systolic_bp_first_inf_100inf7', 'systolic_bp_first_inf_100inf8', 'systolic_bp_first_inf_100inf9', 'systolic_bp_lowest_inf_100inf1', 'systolic_bp_lowest_inf_100inf2', 'systolic_bp_lowest_inf_100inf3', 'systolic_bp_lowest_inf_100inf4', 'systolic_bp_lowest_inf_100inf5', 'systolic_bp_lowest_inf_100inf6', 'systolic_bp_lowest_inf_100inf7', 'systolic_bp_lowest_inf_100inf8', 'systolic_bp_lowest_inf_100inf9', 'diastolic_bp_firstinf1', 'diastolic_bp_firstinf2', 'diastolic_bp_firstinf3', 'diastolic_bp_firstinf4', 'diastolic_bp_firstinf5', 'diastolic_bp_firstinf6', 'diastolic_bp_firstinf7', 'diastolic_bp_firstinf8', 'diastolic_bp_firstinf9', 'diastolic_bp_lowestinf1', 'diastolic_bp_lowestinf2', 'diastolic_bp_lowestinf3', 'diastolic_bp_lowestinf4', 'diastolic_bp_lowestinf5', 'diastolic_bp_lowestinf6', 'diastolic_bp_lowestinf7', 'diastolic_bp_lowestinf8', 'diastolic_bp_lowestinf9', 'respiratory_rate_firstinf1', 'respiratory_rate_firstinf2', 'respiratory_rate_firstinf3', 'respiratory_rate_firstinf4', 'respiratory_rate_firstinf5', 'respiratory_rate_firstinf6', 'respiratory_rate_firstinf7', 'respiratory_rate_firstinf8', 'respiratory_rate_firstinf9', 'respiratory_rate_highestinf1', 'respiratory_rate_highestinf2', 'respiratory_rate_highestinf3', 'respiratory_rate_highestinf4', 'respiratory_rate_highestinf5', 'respiratory_rate_highestinf6', 'respiratory_rate_highestinf7', 'respiratory_rate_highestinf8', 'respiratory_rate_highestinf9', 'gcs_inf_15_firstinf1', 'gcs_inf_15_firstinf2', 'gcs_inf_15_firstinf3', 'gcs_inf_15_firstinf4', 'gcs_inf_15_firstinf5', 'gcs_inf_15_firstinf6', 'gcs_inf_15_firstinf7', 'gcs_inf_15_firstinf8', 'gcs_inf_15_firstinf9', 'gcs_inf_15_lowestinf1', 'gcs_inf_15_lowestinf2', 'gcs_inf_15_lowestinf3', 'gcs_inf_15_lowestinf4', 'gcs_inf_15_lowestinf5', 'gcs_inf_15_lowestinf6', 'gcs_inf_15_lowestinf7', 'gcs_inf_15_lowestinf8', 'gcs_inf_15_lowestinf9', 'lvl_consc_alertinf1', 'lvl_consc_alertinf2', 'lvl_consc_alertinf3', 'lvl_consc_alertinf4', 'lvl_consc_alertinf5', 'lvl_consc_alertinf6', 'lvl_consc_alertinf7', 'lvl_consc_alertinf8', 'lvl_consc_alertinf9', 'temperature_highestinf1', 'temperature_highestinf2', 'temperature_highestinf3', 'temperature_highestinf4', 'temperature_highestinf5', 'temperature_highestinf6', 'temperature_highestinf7', 'temperature_highestinf8', 'temperature_highestinf9', 'temperature_lowestinf1', 'temperature_lowestinf2', 'temperature_lowestinf3', 'temperature_lowestinf4', 'temperature_lowestinf5', 'temperature_lowestinf6', 'temperature_lowestinf7', 'temperature_lowestinf8', 'temperature_lowestinf9', 'temperature_sup_38inf1', 'temperature_sup_38inf2', 'temperature_sup_38inf3', 'temperature_sup_38inf4', 'temperature_sup_38inf5', 'temperature_sup_38inf6', 'temperature_sup_38inf7', 'temperature_sup_38inf8', 'temperature_sup_38inf9', 'temperature_inf_35inf1', 'temperature_inf_35inf2', 'temperature_inf_35inf3', 'temperature_inf_35inf4', 'temperature_inf_35inf5', 'temperature_inf_35inf6', 'temperature_inf_35inf7', 'temperature_inf_35inf8', 'temperature_inf_35inf9', 'spo2_firstinf1', 'spo2_firstinf2', 'spo2_firstinf3', 'spo2_firstinf4', 'spo2_firstinf5', 'spo2_firstinf6', 'spo2_firstinf7', 'spo2_firstinf8', 'spo2_firstinf9', 'spo2_lowestinf1', 'spo2_lowestinf2', 'spo2_lowestinf3', 'spo2_lowestinf4', 'spo2_lowestinf5', 'spo2_lowestinf6', 'spo2_lowestinf7', 'spo2_lowestinf8', 'spo2_lowestinf9', 'o2_gabeinf1', 'o2_gabeinf2', 'o2_gabeinf3', 'o2_gabeinf4', 'o2_gabeinf5', 'o2_gabeinf6', 'o2_gabeinf7', 'o2_gabeinf8', 'o2_gabeinf9', 'THZ (Thrombozyten)inf1', 'THZ (Thrombozyten)inf2', 'THZ (Thrombozyten)inf3', 'THZ (Thrombozyten)inf4', 'THZ (Thrombozyten)inf5', 'THZ (Thrombozyten)inf6', 'THZ (Thrombozyten)inf7', 'THZ (Thrombozyten)inf8', 'THZ (Thrombozyten)inf9', 'NAinf1', 'NAinf2', 'NAinf3', 'NAinf4', 'NAinf5', 'NAinf6', 'NAinf7', 'NAinf8', 'NAinf9', 'ASATinf1', 'ASATinf2', 'ASATinf3', 'ASATinf4', 'ASATinf5', 'ASATinf6', 'ASATinf7', 'ASATinf8', 'ASATinf9', 'UREA = Harnstoffinf1', 'UREA = Harnstoffinf2', 'UREA = Harnstoffinf3', 'UREA = Harnstoffinf4', 'UREA = Harnstoffinf5', 'UREA = Harnstoffinf6', 'UREA = Harnstoffinf7', 'UREA = Harnstoffinf8', 'UREA = Harnstoffinf9', 'GFRinf1', 'GFRinf2', 'GFRinf3', 'GFRinf4', 'GFRinf5', 'GFRinf6', 'GFRinf7', 'GFRinf8', 'GFRinf9', 'PCThsinf1', 'PCThsinf2', 'PCThsinf3', 'PCThsinf4', 'PCThsinf5', 'PCThsinf6', 'PCThsinf7', 'PCThsinf8', 'PCThsinf9', 'EOS =eosinophileinf1', 'EOS =eosinophileinf2', 'EOS =eosinophileinf3', 'EOS =eosinophileinf4', 'EOS =eosinophileinf5', 'EOS =eosinophileinf6', 'EOS =eosinophileinf7', 'EOS =eosinophileinf8', 'EOS =eosinophileinf9', 'CRPinf1', 'CRPinf2', 'CRPinf3', 'CRPinf4', 'CRPinf5', 'CRPinf6', 'CRPinf7', 'CRPinf8', 'CRPinf9', 'KA = Kaliuminf1', 'KA = Kaliuminf2', 'KA = Kaliuminf3', 'KA = Kaliuminf4', 'KA = Kaliuminf5', 'KA = Kaliuminf6', 'KA = Kaliuminf7', 'KA = Kaliuminf8', 'KA = Kaliuminf9', 'Hbinf1', 'Hbinf2', 'Hbinf3', 'Hbinf4', 'Hbinf5', 'Hbinf6', 'Hbinf7', 'Hbinf8', 'Hbinf9', 'ALATinf1', 'ALATinf2', 'ALATinf3', 'ALATinf4', 'ALATinf5', 'ALATinf6', 'ALATinf7', 'ALATinf8', 'ALATinf9', 'INRiHinf1', 'INRiHinf2', 'INRiHinf3', 'INRiHinf4', 'INRiHinf5', 'INRiHinf6', 'INRiHinf7', 'INRiHinf8', 'INRiHinf9', 'CRinf1', 'CRinf2', 'CRinf3', 'CRinf4', 'CRinf5', 'CRinf6', 'CRinf7', 'CRinf8', 'CRinf9', 'pHinf1', 'pHinf2', 'pHinf3', 'pHinf4', 'pHinf5', 'pHinf6', 'pHinf7', 'pHinf8', 'pHinf9', 'LACT =lactatinf1', 'LACT =lactatinf2', 'LACT =lactatinf3', 'LACT =lactatinf4', 'LACT =lactatinf5', 'LACT =lactatinf6', 'LACT =lactatinf7', 'LACT =lactatinf8', 'LACT =lactatinf9', 'GGTinf1', 'GGTinf2', 'GGTinf3', 'GGTinf4', 'GGTinf5', 'GGTinf6', 'GGTinf7', 'GGTinf8', 'GGTinf9', 'BIC_stinf1', 'BIC_stinf2', 'BIC_stinf3', 'BIC_stinf4', 'BIC_stinf5', 'BIC_stinf6', 'BIC_stinf7', 'BIC_stinf8', 'BIC_stinf9', 'BIC_VBinf1', 'BIC_VBinf2', 'BIC_VBinf3', 'BIC_VBinf4', 'BIC_VBinf5', 'BIC_VBinf6', 'BIC_VBinf7', 'BIC_VBinf8', 'BIC_VBinf9', 'Leukinf1', 'Leukinf2', 'Leukinf3', 'Leukinf4', 'Leukinf5', 'Leukinf6', 'Leukinf7', 'Leukinf8', 'Leukinf9', 'leuk_sup_10inf1', 'leuk_sup_10inf2', 'leuk_sup_10inf3', 'leuk_sup_10inf4', 'leuk_sup_10inf5', 'leuk_sup_10inf6', 'leuk_sup_10inf7', 'leuk_sup_10inf8', 'leuk_sup_10inf9', 'leuk_inf_4inf1', 'leuk_inf_4inf2', 'leuk_inf_4inf3', 'leuk_inf_4inf4', 'leuk_inf_4inf5', 'leuk_inf_4inf6', 'leuk_inf_4inf7', 'leuk_inf_4inf8', 'leuk_inf_4inf9', 'leuk_inf_0_5inf1', 'leuk_inf_0_5inf2', 'leuk_inf_0_5inf3', 'leuk_inf_0_5inf4', 'leuk_inf_0_5inf5', 'leuk_inf_0_5inf6', 'leuk_inf_0_5inf7', 'leuk_inf_0_5inf8', 'leuk_inf_0_5inf9', 'NRBCminf1', 'NRBCminf2', 'NRBCminf3', 'NRBCminf4', 'NRBCminf5', 'NRBCminf6', 'NRBCminf7', 'NRBCminf8', 'NRBCminf9', 'NRBCm_10_percentinf1', 'NRBCm_10_percentinf2', 'NRBCm_10_percentinf3', 'NRBCm_10_percentinf4', 'NRBCm_10_percentinf5', 'NRBCm_10_percentinf6', 'NRBCm_10_percentinf7', 'NRBCm_10_percentinf8', 'NRBCm_10_percentinf9', 'AUGENinf1', 'AUGENinf2', 'AUGENinf3', 'AUGENinf4', 'AUGENinf5', 'AUGENinf6', 'AUGENinf7', 'AUGENinf8', 'AUGENinf9', 'CHIRinf1', 'CHIRinf2', 'CHIRinf3', 'CHIRinf4', 'CHIRinf5', 'CHIRinf6', 'CHIRinf7', 'CHIRinf8', 'CHIRinf9', 'FTNinf1', 'FTNinf2', 'FTNinf3', 'FTNinf4', 'FTNinf5', 'FTNinf6', 'FTNinf7', 'FTNinf8', 'FTNinf9', 'HCHinf1', 'HCHinf2', 'HCHinf3', 'HCHinf4', 'HCHinf5', 'HCHinf6', 'HCHinf7', 'HCHinf8', 'HCHinf9', 'HGCinf1', 'HGCinf2', 'HGCinf3', 'HGCinf4', 'HGCinf5', 'HGCinf6', 'HGCinf7', 'HGCinf8', 'HGCinf9', 'HNOinf1', 'HNOinf2', 'HNOinf3', 'HNOinf4', 'HNOinf5', 'HNOinf6', 'HNOinf7', 'HNOinf8', 'HNOinf9', 'KARDinf1', 'KARDinf2', 'KARDinf3', 'KARDinf4', 'KARDinf5', 'KARDinf6', 'KARDinf7', 'KARDinf8', 'KARDinf9', 'KINDinf1', 'KINDinf2', 'KINDinf3', 'KINDinf4', 'KINDinf5', 'KINDinf6', 'KINDinf7', 'KINDinf8', 'KINDinf9', 'MEDinf1', 'MEDinf2', 'MEDinf3', 'MEDinf4', 'MEDinf5', 'MEDinf6', 'MEDinf7', 'MEDinf8', 'MEDinf9', 'NCHinf1', 'NCHinf2', 'NCHinf3', 'NCHinf4', 'NCHinf5', 'NCHinf6', 'NCHinf7', 'NCHinf8', 'NCHinf9', 'NEUROinf1', 'NEUROinf2', 'NEUROinf3', 'NEUROinf4', 'NEUROinf5', 'NEUROinf6', 'NEUROinf7', 'NEUROinf8', 'NEUROinf9', 'ORTHOinf1', 'ORTHOinf2', 'ORTHOinf3', 'ORTHOinf4', 'ORTHOinf5', 'ORTHOinf6', 'ORTHOinf7', 'ORTHOinf8', 'ORTHOinf9', 'PLWSinf1', 'PLWSinf2', 'PLWSinf3', 'PLWSinf4', 'PLWSinf5', 'PLWSinf6', 'PLWSinf7', 'PLWSinf8', 'PLWSinf9', 'PSYCHinf1', 'PSYCHinf2', 'PSYCHinf3', 'PSYCHinf4', 'PSYCHinf5', 'PSYCHinf6', 'PSYCHinf7', 'PSYCHinf8', 'PSYCHinf9', 'SKGinf1', 'SKGinf2', 'SKGinf3', 'SKGinf4', 'SKGinf5', 'SKGinf6', 'SKGinf7', 'SKGinf8', 'SKGinf9', 'SRCHIRinf1', 'SRCHIRinf2', 'SRCHIRinf3', 'SRCHIRinf4', 'SRCHIRinf5', 'SRCHIRinf6', 'SRCHIRinf7', 'SRCHIRinf8', 'SRCHIRinf9', 'SRMEDinf1', 'SRMEDinf2', 'SRMEDinf3', 'SRMEDinf4', 'SRMEDinf5', 'SRMEDinf6', 'SRMEDinf7', 'SRMEDinf8', 'SRMEDinf9', 'TCHinf1', 'TCHinf2', 'TCHinf3', 'TCHinf4', 'TCHinf5', 'TCHinf6', 'TCHinf7', 'TCHinf8', 'TCHinf9', 'UROinf1', 'UROinf2', 'UROinf3', 'UROinf4', 'UROinf5', 'UROinf6', 'UROinf7', 'UROinf8', 'UROinf9', 'VCHinf1', 'VCHinf2', 'VCHinf3', 'VCHinf4', 'VCHinf5', 'VCHinf6', 'VCHinf7', 'VCHinf8', 'VCHinf9', 'Ambulanzinf1', 'Ambulanzinf2', 'Ambulanzinf3', 'Ambulanzinf4', 'Ambulanzinf5', 'Ambulanzinf6', 'Ambulanzinf7', 'Ambulanzinf8', 'Ambulanzinf9', 'City-Notfallinf1', 'City-Notfallinf2', 'City-Notfallinf3', 'City-Notfallinf4', 'City-Notfallinf5', 'City-Notfallinf6', 'City-Notfallinf7', 'City-Notfallinf8', 'City-Notfallinf9', 'Externes Spital (Ambulanz)inf1', 'Externes Spital (Ambulanz)inf2', 'Externes Spital (Ambulanz)inf3', 'Externes Spital (Ambulanz)inf4', 'Externes Spital (Ambulanz)inf5', 'Externes Spital (Ambulanz)inf6', 'Externes Spital (Ambulanz)inf7', 'Externes Spital (Ambulanz)inf8', 'Externes Spital (Ambulanz)inf9', 'Externes Spital (privat)inf1', 'Externes Spital (privat)inf2', 'Externes Spital (privat)inf3', 'Externes Spital (privat)inf4', 'Externes Spital (privat)inf5', 'Externes Spital (privat)inf6', 'Externes Spital (privat)inf7', 'Externes Spital (privat)inf8', 'Externes Spital (privat)inf9', 'Hausarzt (Ambulanz)inf1', 'Hausarzt (Ambulanz)inf2', 'Hausarzt (Ambulanz)inf3', 'Hausarzt (Ambulanz)inf4', 'Hausarzt (Ambulanz)inf5', 'Hausarzt (Ambulanz)inf6', 'Hausarzt (Ambulanz)inf7', 'Hausarzt (Ambulanz)inf8', 'Hausarzt (Ambulanz)inf9', 'Hausarzt (privat)inf1', 'Hausarzt (privat)inf2', 'Hausarzt (privat)inf3', 'Hausarzt (privat)inf4', 'Hausarzt (privat)inf5', 'Hausarzt (privat)inf6', 'Hausarzt (privat)inf7', 'Hausarzt (privat)inf8', 'Hausarzt (privat)inf9', 'Inselklinikinf1', 'Inselklinikinf2', 'Inselklinikinf3', 'Inselklinikinf4', 'Inselklinikinf5', 'Inselklinikinf6', 'Inselklinikinf7', 'Inselklinikinf8', 'Inselklinikinf9', 'Polizei (In Begleitung)inf1', 'Polizei (In Begleitung)inf2', 'Polizei (In Begleitung)inf3', 'Polizei (In Begleitung)inf4', 'Polizei (In Begleitung)inf5', 'Polizei (In Begleitung)inf6', 'Polizei (In Begleitung)inf7', 'Polizei (In Begleitung)inf8', 'Polizei (In Begleitung)inf9', 'Polizei (Strafanstalten)inf1', 'Polizei (Strafanstalten)inf2', 'Polizei (Strafanstalten)inf3', 'Polizei (Strafanstalten)inf4', 'Polizei (Strafanstalten)inf5', 'Polizei (Strafanstalten)inf6', 'Polizei (Strafanstalten)inf7', 'Polizei (Strafanstalten)inf8', 'Polizei (Strafanstalten)inf9', 'Psychiatrieinf1', 'Psychiatrieinf2', 'Psychiatrieinf3', 'Psychiatrieinf4', 'Psychiatrieinf5', 'Psychiatrieinf6', 'Psychiatrieinf7', 'Psychiatrieinf8', 'Psychiatrieinf9', 'Rega (Externes Spital)inf1', 'Rega (Externes Spital)inf2', 'Rega (Externes Spital)inf3', 'Rega (Externes Spital)inf4', 'Rega (Externes Spital)inf5', 'Rega (Externes Spital)inf6', 'Rega (Externes Spital)inf7', 'Rega (Externes Spital)inf8', 'Rega (Externes Spital)inf9', 'Rega (Primär)inf1', 'Rega (Primär)inf2', 'Rega (Primär)inf3', 'Rega (Primär)inf4', 'Rega (Primär)inf5', 'Rega (Primär)inf6', 'Rega (Primär)inf7', 'Rega (Primär)inf8', 'Rega (Primär)inf9', 'Rega (Repatriierung)inf1', 'Rega (Repatriierung)inf2', 'Rega (Repatriierung)inf3', 'Rega (Repatriierung)inf4', 'Rega (Repatriierung)inf5', 'Rega (Repatriierung)inf6', 'Rega (Repatriierung)inf7', 'Rega (Repatriierung)inf8', 'Rega (Repatriierung)inf9', 'Selbsteinweisunginf1', 'Selbsteinweisunginf2', 'Selbsteinweisunginf3', 'Selbsteinweisunginf4', 'Selbsteinweisunginf5', 'Selbsteinweisunginf6', 'Selbsteinweisunginf7', 'Selbsteinweisunginf8', 'Selbsteinweisunginf9', 'Sonstigesinf1', 'Sonstigesinf2', 'Sonstigesinf3', 'Sonstigesinf4', 'Sonstigesinf5', 'Sonstigesinf6', 'Sonstigesinf7', 'Sonstigesinf8', 'Sonstigesinf9', 'UNZ Ambulatoriuminf1', 'UNZ Ambulatoriuminf2', 'UNZ Ambulatoriuminf3', 'UNZ Ambulatoriuminf4', 'UNZ Ambulatoriuminf5', 'UNZ Ambulatoriuminf6', 'UNZ Ambulatoriuminf7', 'UNZ Ambulatoriuminf8', 'UNZ Ambulatoriuminf9']


# cols = ["LACT =lactat", "LACT =lactatinf7", "CRP", "GGT", "GGTinf5", "INRiHinf5", "INRiH", "THZ (Thrombozyten)", "THZ (Thrombozyten)inf3", "ASATinf3", "UREA = Harnstoff", "UREA = Harnstoffinf3", "ort_vor_aufnahme_spitalinf2", "BIC_VB",  "BIC_st",  "BIC_stinf1", "lvl_consc_alert", "pHinf1", "CRinf8", "EOS =eosinophileinf6", "age_admission_unz", "diastolic_bp_firstinf4", "respiratory_rate_firstinf3", "systolic_bp_firstinf4", "Hb", "PCThs"]
# print(len(cols))

df = pd.read_csv("x.csv")
df = df.join(pd.get_dummies(df["admission_choice"]))
df = df.join(pd.get_dummies(df["referral_unz"]))
# df = df.join(pd.get_dummies(df["ort_vor_aufnahme_spital"]))
# df = df.join(pd.get_dummies(df["triage"]))
df.referral_unz = pd.factorize(df.referral_unz)[0]
df.admission_choice = pd.factorize(df.admission_choice)[0]
df.ort_vor_aufnahme_spital = pd.factorize(df.ort_vor_aufnahme_spital)[0]
df = df.select_dtypes(['number'])

cols3 = ["age_admission_unz", "sex", "triage",
                 "pulse_first", "frequency_first", "systolic_bp_first",
                 "diastolic_bp_first", "respiratory_rate_first",
                 "lvl_consc_alert", "temperature_highest", "temperature_lowest", "spo2_first",
                 "o2_gabe", "THZ (Thrombozyten)", "NA", "ASAT", "UREA = Harnstoff", "GFR",
                 "PCThs", "EOS =eosinophile", "CRP", "KA = Kalium", "Hb", "ALAT", "INRiH",
                 "CR", "pH", "LACT =lactat", "GGT", "BIC_st", "BIC_VB", "Leuk"]



for i in df.columns:
  for j in range(1,10):
    df[i+"inf"+str(j)] = (df[i] >= df[i].quantile(0.1*j)).astype(int) 

for i in cols3:
  for j in cols3:
    df[i+"div"+j] = df[i]/df[j]
    df[i+"mul"+j] = df[i]*df[j]

def z_score(df):
    # copy the dataframe
    df_std = df.copy()
    # apply the z-score method
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
        
    return df_std

from keras import backend as K
import tensorflow as tf

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def calc_f1_score(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return (2*((precision*recall)/(precision+recall+K.epsilon()))).numpy()

def get_f1(y_test, y_pred, move=0.0):
  y_pred = pd.DataFrame(y_pred)
  y_pred = round(y_pred+move)
  y_pred = tf.cast(y_pred, tf.float32)
  y_test = tf.cast(y_test, tf.float32)
  return calc_f1_score(y_test, y_pred)

cols = ["age_admission_unz", "triage", "referral_unz", "admission_choice", "diastolic_bp_first", 
"respiratory_rate_first", "gcs_inf_15_first", "lvl_consc_alert", 
"temperature_lowest", "spo2_first", "THZ (Thrombozyten)", "NA", 
"EOS =eosinophile", "CRP", "KA = Kalium", 
"Hb", "INRiH", "CR", "LACT =lactat", "GGT", "BIC_st", "leuk_sup_10", 
"leuk_inf_4", "leuk_inf_0_5", "LACT =lactatinf7", "GGTinf5", "INRiHinf5", 
"THZ (Thrombozyten)inf3", "ASATinf3", "UREA = Harnstoffinf3", 
"ort_vor_aufnahme_spitalinf2", "pHinf1", 
"EOS =eosinophileinf6","diastolic_bp_firstinf4", "respiratory_rate_firstinf3", 
"systolic_bp_firstinf4", "PLWS", 'Hausarzt (privat)',
'Inselklinik', 'Polizei (In Begleitung)', 'temperature_lowestinf9', 'EOS =eosinophileinf8', 'frequency_firstinf9', 'GFRdivEOS =eosinophile', 'EOS =eosinophiledivrespiratory_rate_first', 'LACT =lactatmultemperature_lowest']
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn import tree
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import fbeta_score


!pip install fasttrees
from fasttrees.fasttrees import FastFrugalTreeClassifier
# 'EOS =eosinophiledivCR' 2
# Augen 3
import random

def minmaxint(perc, x):
    return int(round(min(1,max(0,x)),0))


def NN():

  model.add(Bidirectional(LSTM(64)))


j = 0
for i in [1]:
  j+=1
  # i = random.choice(df.columns)
  best_cols = cols.copy()
  # best_cols.append(i)

  # best_cols.append("blood_culture_positive")
  # best_cols.append("pseudoid_patient")

  print(len(df))
  # df.drop(df[df.sex == 1].index, inplace=True)
  # df.drop(df[df.age_admission_unz < 18].index, inplace=True)
  # df['mort_28_days'] = df['mort_28_days'].fillna(2)
  # df.drop(df[abs(df.mort_28_days) > 1].index, inplace=True)
  print(len(df))

  proc_x = df[best_cols]
  
  proc_y = df["blood_culture_positive"]
  # proc_y = df["mort_28_days"].astype(int)
  group_by = df["pseudoid_patient"]
  proc_x = proc_x.where(proc_x>0.09, other=0)
  proc_x = proc_x.where(proc_x<10000, other=10000)
  proc_x.to_csv('sepsis_prediction.csv', index=False)
  proc_x = proc_x*100
  
  from sklearn.manifold import TSNE
  import seaborn as sns
  
  tsne = TSNE(n_components=2, verbose=1, random_state=123)
  z = tsne.fit_transform(proc_x) 
  
  df = pd.DataFrame()
  df["y"] = proc_y
  df["Component 1"] = z[:,0]
  df["Component 2"] = z[:,1]

  plot = sns.scatterplot(x="Component 1", y="Component 2", hue=df.y.tolist(),
                  palette=sns.color_palette("hls", 2),
                  data=df).set(title="T-SNE - Positive Blood Culture Prediction") 

  # new_labels = ['label 1', 'label 2']
  #for t, l in zip(plot._legend.texts, new_labels): 
  #  t.set_text(l)
  # print(plot.legend_elements())
  # plt.legend(plot.legend_elements()[0], ['Positive', 'Negative'], loc='upper right')
  plt.savefig("outputa.png")
  # proc_x = proc_x.fillna(0)
  
  kf = GroupKFold(n_splits=8)
  roc_auc_list = []
  f1_list = []
  roc_auc_list_int = []
  recall_list = []
  precision_list = []
  accuracy_list = []

  
    # adj = 0.03
    # return int(round(min(1,max(0,x-perc+0.5+adj)),0))

  mean_fpr = np.linspace(0, 1, 100)
  tprs = []
  aucs = []
  fig, ax = plt.subplots()
  # fig2, ax2 = plt.subplots()

  count = 1
  for train_index, test_index in kf.split(proc_x, proc_y, groups=group_by):
    X_train, X_test = proc_x.iloc[train_index], proc_x.iloc[test_index]
    y_train, y_test = proc_y.iloc[train_index], proc_y.iloc[test_index]
    bin_count = np.bincount(y_train.values.flatten())
    class_weight = bin_count[0]/bin_count[1]
    # clf = Ridge()
    clf = LinearRegression()
    # clf = RidgeClassifier()
    # clf = FastFrugalTreeClassifier()
    # clf = xgb.XGBRegressor(objective="reg:squarederror", random_state=0, scale_pos_weight=4.8)
    clf.fit(X_train, y_train, sample_weight=(class_weight-1)*y_train.values.flatten()+1)
    y_pred = clf.predict(X_test)
    perc = np.percentile(y_pred, 80)
    y_pred_int = list(map(minmaxint, [perc]*len(y_pred), y_pred))
    # y_pred_int = list(map(round, y_pred))
    roc_auc_list.append(roc_auc_score(y_test, y_pred))
    roc_auc_list_int.append(roc_auc_score(y_test, y_pred_int))
    
    f1_list.append(f1_score(y_test, y_pred_int))
    q= fbeta_score(y_test, y_pred_int, average='binary', beta=0.5)
    print(q)
    recall_list.append(recall_score(y_test, y_pred_int))
    precision_list.append(precision_score(y_test, y_pred_int))
    accuracy_list.append(accuracy_score(y_test, y_pred_int))
    
    viz = plot_roc_curve(clf, X_test, y_test, name='ROC fold {}'.format(count),
                         alpha=0.3, lw=1, ax=ax)  
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    
    # viz = plot_precision_recall_curve(clf, X_test, y_test, name='Recall/Precision fold {}'.format(count),
    #                     alpha=0.3, lw=1, ax=ax2) 
     
    count+=1

  
    
    
  print("NEW:", np.mean(roc_auc_list), np.mean(roc_auc_list_int), np.mean(f1_list), np.mean(recall_list), np.mean(precision_list), np.mean(accuracy_list))
  print("OLD: 0.7352555928735807 0.69889741075676 0.44061706321273053")
  # Random ROC 0.5 and F1 0.29
  if np.mean(roc_auc_list)> 0.7352 and np.mean(roc_auc_list_int) > 0.6989 and np.mean(f1_list) > 0.4407:
    print("NEW HIGHSCORE")
    print("####################################")
    print(i)  
    print("NEW:", np.mean(roc_auc_list), np.mean(roc_auc_list_int), np.mean(f1_list))
    print("OLD: 0.7352555928735807 0.69889741075676 0.44061706321273053")
    # import sys
    #sys.exit()
  
 
  ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
          label='Chance', alpha=.8)

  mean_tpr = np.mean(tprs, axis=0)
  mean_tpr[-1] = 1.0
  mean_auc = auc(mean_fpr, mean_tpr)
  std_auc = np.std(aucs)
  ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

  std_tpr = np.std(tprs, axis=0)
  tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
  tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
  ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                  label=r'$\pm$ 1 std. dev.')

  ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title="ROC Curve - Positive Blood Culture Prediction")
  ax2.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title="Precision / Recall - Positive Blood Culture Prediction")
  ax.legend(loc="lower right", prop={'size': 7})
  
  ax2.legend(loc="upper right", prop={'size': 7})
  ax2.axhline(0.2, color='r', linestyle='--')
  ax2.plot(0.695, 0.322, color='r', marker='o', markersize=10)
  
  plt.savefig('sepsisprediction.png', dpi=300)
  plt.show()



to_predict = "blood_culture_positive"
imp = pd.read_csv("x.csv")
imp.entrance_unz_date = pd.to_datetime(imp.entrance_unz_date)
imp.ort_vor_aufnahme_spital = pd.factorize(imp.ort_vor_aufnahme_spital)[0]
imp.referral_unz = pd.factorize(imp.referral_unz)[0]
imp.admission_choice = pd.factorize(imp.admission_choice)[0]
imp_x = imp.sort_values(by='entrance_unz_date')
imp_x.entrance_unz_date = pd.to_datetime(imp_x.entrance_unz_date).dt.dayofyear
imp_x.dropna(axis=0, subset=[to_predict], inplace=True)

features = [to_predict, "pseudoid_patient", "age_admission_unz", "sex", "triage", "referral_unz", "ort_vor_aufnahme_spital", "time_unz", "admission_choice",
                 "pulse_first", "frequency_first", "systolic_bp_first",
                 "diastolic_bp_first", "respiratory_rate_first", "gcs_inf_15_first",
                 "lvl_consc_alert", "temperature_highest", "temperature_lowest", "spo2_first",
                 "o2_gabe", "THZ (Thrombozyten)", "NA", "ASAT", "UREA = Harnstoff", "GFR",
                 "PCThs", "EOS =eosinophile", "CRP", "KA = Kalium", "Hb", "ALAT", "INRiH",
                 "CR", "pH", "LACT =lactat", "GGT", "BIC_st", "BIC_VB", "Leuk", "entrance_unz_date"]

to_check = imp_x[features]
for i in to_check.columns[1:]:
  for j in range(1,10):
    to_check[i+"inf"+str(j)] = (to_check[i] >= to_check[i].quantile(0.1*j)).astype(int)
    # print(to_check[i+"inf"+str(j)])

imp_x = to_check

to_check = to_check[[to_predict, "pseudoid_patient", "LACT =lactat", "LACT =lactatinf7", "CRP", "GGT", "GGTinf5", "INRiHinf5", "INRiH", "THZ (Thrombozyten)", "THZ (Thrombozyten)inf3", "ASATinf3", "UREA = Harnstoff", "UREA = Harnstoffinf3", "ort_vor_aufnahme_spitalinf2", "BIC_VB",  "BIC_st",  "BIC_stinf1", "lvl_consc_alert",
                  "pHinf1", "CRinf8", "EOS =eosinophileinf6", "age_admission_unz", "diastolic_bp_firstinf4", "respiratory_rate_firstinf3", "systolic_bp_firstinf4", "Hb", "PCThs"]]

features_better2 = [to_predict, "LACT =lactat", "LACT =lactatinf7", "CRP", "GGT", "GGTinf5", "INRiHinf5", "INRiH", "THZ (Thrombozyten)", "THZ (Thrombozyten)inf3", "ASATinf3", "UREA = Harnstoff", "UREA = Harnstoffinf3", "ort_vor_aufnahme_spitalinf2", "BIC_VB",  "BIC_st",  "BIC_stinf1", "lvl_consc_alert",
                   "pHinf1", "CRinf8", "EOS =eosinophileinf6", "age_admission_unz", "diastolic_bp_firstinf4", "respiratory_rate_firstinf3", "systolic_bp_firstinf4", "Hb", "PCThs"]
feature_list = [features_better2]

from sklearn.model_selection import train_test_split

proc_y = imp_x[[to_predict]]
proc_x = imp_x[features_better2]
proc_x = proc_x.fillna(-1)
proc_x = z_score(proc_x)
proc_x = proc_x[proc_x.columns[2:]]
X_train, X_test, y_train, y_test = train_test_split(proc_x, proc_y, test_size=0.2, shuffle=False)
bin_count = np.bincount(y_train.values.flatten())
class_weight = bin_count[0]/bin_count[1]
kf = GroupKFold(n_splits=8)
roc_auc_list = []
f1_list = []
for train_index, test_index in kf.split(proc_x, proc_y, groups=imp_x["pseudoid_patient"]):
  clf = RidgeClassifier(class_weight='balanced') 
  X_trains, X_tests = proc_x.iloc[train_index], proc_x.iloc[test_index]
  y_trains, y_tests = proc_y.iloc[train_index], proc_y.iloc[test_index]
  clf.fit(X_trains, y_trains.values.flatten())
  y_preds = list(map(round, clf.predict(X_tests)))
  roc_auc_list.append(roc_auc_score(y_tests, y_preds, sample_weight=None))
  f1_list.append(f1_score(y_tests, y_preds))
print(clf, "GrAnd Total:", np.mean(roc_auc_list), np.mean(f1_list))
