df = df.where(df>0.09, other=0)
df = df.where(df<10000, other=10000)
	
target = "blood_culture_positive"
corr_list = []
for i in FEATURE_NAMES:
	corr_list.append((i, round(df[[i, "blood_culture_positive"]].corr().values[0][1],3),
round(df.loc[(df[target]==1),i].mean(),3), 
round(df.loc[(df[target]==1),i].std(), 3), 
round(df.loc[(df[target]==0),i].mean(), 3), 
round(df.loc[(df[target]==0),i].std(), 3)))

corr_list.sort(key=lambda a: abs(a[1]))
print(f"{'Feature':<32} | {'Corr.':<8} | {'Pos.Mean':<8} {'Pos.Std':<8} | {'Neg.Mean':<8} {'Neg.Std':<8}")
for i in corr_list:
	print("{0:<32} | {1:<8} | {2:<8} {3:<8} | {4:<8} {5:<8}".format(i[0][:min(30,len(i[0]))], i[1], i[2],i[3],i[4],i[5]))
