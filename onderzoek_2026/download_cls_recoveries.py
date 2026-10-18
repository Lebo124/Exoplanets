from pathlib import Path
import urllib.request,concurrent.futures,json,hashlib
import pandas as pd
from datetime import datetime,timezone
root=Path(__file__).resolve().parent/'detectability_sources';out=root/'recoveries';out.mkdir(exist_ok=True)
a=pd.read_csv(root/'group4_CLS_crossmatch.csv',keep_default_na=False);a=a[a.CLS_matches.ne('')]
def get(row):
 name=row.CLS_matches;p=out/(name+'_recoveries.csv');u='https://raw.githubusercontent.com/leerosenthalj/CLSI/master/completeness/recoveries_all/'+p.name
 try:
  with urllib.request.urlopen(u,timeout=50) as r:raw=r.read()
  p.write_bytes(raw)
  return {'hostname':row.hostname,'cls_name':name,'url':u,'file':str(p.relative_to(root)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 except Exception as e:return {'hostname':row.hostname,'cls_name':name,'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(10) as pool:
 results=list(pool.map(get,a.itertuples(index=False)))
(root/'download_provenance.json').write_text(json.dumps({'downloaded_utc':datetime.now(timezone.utc).isoformat(),'source':'Rosenthal et al. 2021 California Legacy Survey I, public per-star injection-recovery tests','repository_tree_sha':json.loads((root/'CLSI_tree.json').read_text())['sha'],'files':results},indent=2))
print('Successful:',sum('error' not in r for r in results),'/',len(results),flush=True)
print([r for r in results if 'error' in r],flush=True)
