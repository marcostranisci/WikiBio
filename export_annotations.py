### separa in frasi
'''
l = list()
     ...: i = 0
     ...: for idx,t in enumerate(df.token.values):
     ...:     if df.token.values[idx-1]=='-' and df.token.values[idx]=='-':
     ...:         l.append(i)
     ...:         i+=1
     ...:     elif df.token.values[idx]=='-' and df.token.values[idx-1]!='-':
     ...:         l.append(i)
     ...:     else: l.append(i)
'''

### toglie i caratteri in più
'''In [160]: l = list()
     ...: for idx,tok in enumerate(df.token.values):
     ...:
     ...:     if tok == '-':
     ...:         if df.token.values[idx-1]=='-':
     ...:             print(df.token.values[idx-1])
     ...:             l.append(idx)
     ...:             l.append(idx-1)
     ...:
'''

# esporta solo le frasi con eventi
'''l = list()
     ...: for item in glob.glob('*.csv'):
     ...:     name = item[:-4]
     ...:     df = pd.read_csv(item)
     ...:     df = df.groupby('sentence').aggregate({'token':list,'label':list})
     ...:     df = df.reset_index()
     ...:     for row in df.iloc[:].values:
     ...:         if 'EVENT' in row[-1]:
     ...:             l.append((name,row[0],row[1]))
'''
