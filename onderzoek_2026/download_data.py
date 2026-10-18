"""Download a dated NASA snapshot; preserve raw values and provenance."""
from pathlib import Path
import urllib.request, urllib.parse, json, hashlib
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parent
features=['pl_bmasse','pl_rade','pl_dens','pl_insol','pl_eqt','pl_orbper','st_mass','st_rad','st_teff','st_lum']
cols=['pl_name','hostname','discoverymethod','disc_year','pl_bmassprov','pl_controv_flag']
for c in features:
    cols += [c,c+'lim',c+'_reflink']
cols += ['pl_bmasseerr1','pl_bmasseerr2','pl_radeerr1','pl_radeerr2']
query='select '+','.join(cols)+' from pscomppars order by pl_name'
url='https://exoplanetarchive.ipac.caltech.edu/TAP/sync?'+urllib.parse.urlencode({'query':query,'format':'csv'})
print('Downloading NASA PSCompPars...',flush=True)
with urllib.request.urlopen(url,timeout=120) as response:
    raw=response.read()
if not raw.startswith(b'pl_name,'): raise RuntimeError(raw[:1000])
p=ROOT/'data/nasa_pscomppars.csv'; p.write_bytes(raw)
meta={'retrieved_utc':datetime.now(timezone.utc).isoformat(),'source_url':url,'query':query,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'documentation':['https://exoplanetarchive.ipac.caltech.edu/docs/pscp_about.html','https://exoplanetarchive.ipac.caltech.edu/docs/pscp_calc.html','https://exoplanetarchive.ipac.caltech.edu/docs/API_TD_columns.html']}
(ROOT/'data/provenance.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2),flush=True)
