df.loc[df['bact_name'] == 'koagulase-negative Staphylokokken', 'blood_culture_positive'] = 0
df.loc[df['bact_name'] == 'Corynebacterium species', 'blood_culture_positive'] = 0
df.loc[df['bact_name'] == 'Bacillus species', 'blood_culture_positive'] = 0
df.loc[df['bact_name'] == 'clostridium perfringens', 'blood_culture_positive'] = 0
df.loc[df['bact_name'] == 'Propionibacterium acnes', 'blood_culture_positive'] = 0
df.loc[df['bact_name'] == 'Clostridium perfringens', 'blood_culture_positive'] = 0
df.loc[df['bact_name'] == 'Bacillus cereus/thuringiensis', 'blood_culture_positive'] = 0 
df.loc[df['bact_name'] == 'Bacillus licheniformis', 'blood_culture_positive'] = 0
