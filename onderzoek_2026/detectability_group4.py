"""Evaluate published per-star injection recoveries for a defined close-in giant benchmark.
No refitting, no universal non-detection claim, no assumed sensitivity from Nobs alone.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis import ROOT
SOURCE=ROOT/'detectability_sources';OUT=ROOT/'results_detectability_group4';MJ=317.828

def wilson(k,n):
    if not n:return np.nan,np.nan
    p=k/n;z=1.959963984540054;den=1+z*z/n
    middle=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return middle-half,middle+half

def analyze():
    OUT.mkdir(exist_ok=True)
    group=pd.read_csv(ROOT/'results_pca/planeten_met_groepen_pca.csv')
    anchors=group[group.SOM_groep.eq(4)]
    cross=pd.read_csv(SOURCE/'group4_CLS_crossmatch.csv',keep_default_na=False)
    obs=pd.read_csv(SOURCE/'observation_stats.csv',dtype={'name':str}).set_index('name')
    rows=[]
    for row in cross.itertuples(index=False):
        rec={'ster':row.hostname,'groep4_planeten':row.group4_planets,'CLS_naam':row.CLS_matches,'bron_getest':False,'testsignalen':0,'teruggevonden':0,'terugvindfractie':np.nan,'wilson95_laag':np.nan,'wilson95_hoog':np.nan,'N_metingen':np.nan,'baseline_jaren':np.nan,'sterke_aanwijzing_90pct':False,'conservatief_90pct':False,'groep1_metgezel_bekend':bool(group[group.hostname.eq(row.hostname)].SOM_groep.eq(1).any())}
        if row.CLS_matches:
            p=SOURCE/'recoveries'/(row.CLS_matches+'_recoveries.csv')
            if p.exists():
                d=pd.read_csv(p)
                assert d.recovered.isin([True,False]).all()
                # Independent benchmark only; these signals are not reclassified via the SOM.
                chosen=d[d.inj_period.between(2,20)&d.inj_msini.between(.3*MJ,3*MJ)]
                n=len(chosen);k=int(chosen.recovered.sum());lo,hi=wilson(k,n)
                rec.update(bron_getest=True,testsignalen=n,teruggevonden=k,terugvindfractie=k/n if n else np.nan,wilson95_laag=lo,wilson95_hoog=hi,sterke_aanwijzing_90pct=bool(n>=20 and k/n>=.9),conservatief_90pct=bool(n>=20 and lo>=.9))
                if row.CLS_matches in obs.index:
                    stat=obs.loc[row.CLS_matches];rec['N_metingen']=stat.Nobs;rec['baseline_jaren']=stat.baseline/365.25
                selected=chosen.assign(ster=row.hostname,CLS_naam=row.CLS_matches)
                selected.to_csv(OUT/(row.CLS_matches+'_geselecteerde_tests.csv'),index=False)
        rows.append(rec)
    table=pd.DataFrame(rows)
    table.to_csv(OUT/'detectiekans_alle_groep4_sterren.csv',index=False)
    tested=table[table.bron_getest].copy();tested.to_csv(OUT/'geteste_sterren.csv',index=False)
    strong=table[table.sterke_aanwijzing_90pct].copy();strong.to_csv(OUT/'sterke_aanwijzingen.csv',index=False)
    cautious=table[table.conservatief_90pct].copy();cautious.to_csv(OUT/'conservatieve_selectie.csv',index=False)
    testedplanet=anchors[anchors.hostname.isin(tested.ster)]
    # All names are unique; percentages count stars, not duplicated multi-planet hosts.
    assert table.ster.is_unique and len(table)==anchors.hostname.nunique()
    result={'group4_planets':len(anchors),'group4_host_stars':anchors.hostname.nunique(),'discovery_methods_planet_counts':anchors.discoverymethod.value_counts().to_dict(),'tested_hosts':len(tested),'tested_group4_planets':len(testedplanet),'not_assessed_hosts':int((~table.bron_getest).sum()),'strong_hosts':len(strong),'conservative_hosts':len(cautious),'strong_hosts_without_group1_known':int((~strong.groep1_metgezel_bekend).sum()),'conservative_hosts_without_group1_known':int((~cautious.groep1_metgezel_bekend).sum()),'tested_injections':int(tested.testsignalen.sum()),'recovered_injections':int(tested.teruggevonden.sum()),'benchmark':{'minimum_mass_MJ':[.3,3],'period_days':[2,20],'minimum_tests':20,'strong_point_estimate_threshold':.9,'conservative_Wilson95_lower_threshold':.9,'eccentricities':'Use original published Kipping beta-distribution injections; not reweighted.'},'important_limit':'A recovery fraction is averaged over finite published signals in this bin, not a guarantee for every individual mass, period, orbital phase, eccentricity or inclination; sin(i) is part of the benchmark minimum mass. Tests use the 2021 published survey dataset, not all 2026 observations. 204 not-assessed stars do not imply inability to detect.','sources':['https://doi.org/10.3847/1538-4365/abe23c','https://github.com/leerosenthalj/CLSI/tree/master/completeness/recoveries_all','https://github.com/leerosenthalj/CLSII/tree/master/legacy_tables'],'model_sha256':hashlib.sha256((ROOT/'results_pca/som_model_pca.npz').read_bytes()).hexdigest()}
    (OUT/'samenvatting.json').write_text(json.dumps(result,indent=2))
    fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained')
    examples=['47 UMa','55 Cnc','14 Her','HD 154345','HD 190360','HD 105618','eps Eri','HR 5183']
    t=tested.set_index('ster').reindex(examples).dropna(subset=['terugvindfractie']).iloc[::-1]
    y=np.arange(len(t));means=t.terugvindfractie.values*100
    ax.errorbar(means,y,xerr=np.maximum(0,np.vstack([means-t.wilson95_laag.values*100,t.wilson95_hoog.values*100-means])),fmt='o',color='#286c8e',capsize=4)
    for i,(name,row) in enumerate(t.iterrows()):ax.text(103,i,f'{int(row.teruggevonden)}/{int(row.testsignalen)}',va='center')
    ax.axvline(90,color='#ae741c',linestyle='--',label='90%')
    ax.set(yticks=y,yticklabels=t.index,xlim=(0,115),xticks=[0,25,50,75,100],xlabel='Teruggevonden testsignalen (%) met 95%-Wilsoninterval',title='Hoe goed waren nabije reuzen detecteerbaar?\n0,3–3 Jupitermassa’s (M sin i), omlooptijd 2–20 dagen')
    ax.spines[['top','right']].set_visible(False);ax.legend(loc='lower left')
    fig.savefig(OUT/'detectiekans_voorbeelden.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2));print(tested[['ster','testsignalen','teruggevonden','terugvindfractie','wilson95_laag','N_metingen','baseline_jaren','conservatief_90pct','groep1_metgezel_bekend']].sort_values('ster').round(4).to_string(index=False))
    return result

if __name__=='__main__':analyze()
