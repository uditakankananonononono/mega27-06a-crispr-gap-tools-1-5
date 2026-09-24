import pandas as pd
l1 = pd.read_excel('data/raw/pridict2/moesm5.xlsx', sheet_name='Library_1')
l1.to_parquet('/tmp/pridict2_l1.parquet')
print('L1 done', l1.shape, flush=True)
pc = pd.read_excel('data/raw/pe_chromatin/moesm3.xlsx', sheet_name='2')
pc.to_parquet('/tmp/pe_chromatin.parquet')
print('PC done', pc.shape, flush=True)
