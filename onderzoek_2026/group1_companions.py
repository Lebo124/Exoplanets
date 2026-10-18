"""Known companions around host stars with a planet in the frozen PCA-SOM group 1.
No training or new classification. Gas-giant proxies and sensitivity are explicit.
"""
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis import ROOT
OUT=ROOT/'results_group1_companions'
MJ=317.828

def analyze():
    OUT.mkdir(exist_ok=True)
    raw=pd.read_csv(ROOT/'data/nasa_pscomppars.csv')
    classified=pd.read_csv(ROOT/'results_pca/planeten_met_groepen_pca.csv')
    assert raw.pl_name.is_unique and classified.pl_name.is_unique
    anchors=classified[classified.SOM_groep.eq(1)].copy()
    hosts=sorted(anchors.hostname.unique())
    # Join the entire snapshot, not just planets with all ten SOM features.
    planets=raw[raw.hostname.isin(hosts)&raw.pl_controv_flag.eq(0)].copy()
    planets['SOM_groep']=planets.pl_name.map(classified.set_index('pl_name').SOM_groep)
    planets['massa_niet_ingevuld']=~planets.pl_bmasse_reflink.str.contains('calculated',case=False,na=False)&~planets.pl_bmassprov.eq('M-R relationship')&planets.pl_bmasse.notna()
    planets['straal_niet_ingevuld']=~planets.pl_rade_reflink.str.contains('calculated',case=False,na=False)&planets.pl_rade.notna()
    # A lower mass/radius bound exceeding the cutoff qualifies; an upper bound does not.
    planets['bruikbare_massagrens']=planets.pl_bmasselim.fillna(0).le(0)&planets.massa_niet_ingevuld
    planets['bruikbare_straalgrens']=planets.pl_radelim.fillna(0).le(0)&planets.straal_niet_ingevuld
    planets['zwaar_13MJ_of_meer']=planets.bruikbare_massagrens&planets.pl_bmasse.ge(13*MJ)
    planets['reus_proxy']=((planets.bruikbare_massagrens&planets.pl_bmasse.ge(50))|(planets.bruikbare_straalgrens&planets.pl_rade.ge(8)))&~planets.zwaar_13MJ_of_meer
    planets['massa_Jupiter']=planets.pl_bmasse/MJ
    records=[]
    for _,anchor in anchors.iterrows():
        others=planets[planets.hostname.eq(anchor.hostname)&planets.pl_name.ne(anchor.pl_name)]
        for _,other in others.iterrows():
            records.append({'ster':anchor.hostname,'groep1_planeet':anchor.pl_name,'groep1_omlooptijd_dagen':anchor.pl_orbper,'begeleider':other.pl_name,'begeleider_groep':other.SOM_groep,'begeleider_omlooptijd_dagen':other.pl_orbper,'begeleider_massa_aarde':other.pl_bmasse,'begeleider_massa_Jupiter':other.massa_Jupiter,'massatype':other.pl_bmassprov,'massalimietvlag':other.pl_bmasselim,'straal_aarde':other.pl_rade,'massa_niet_ingevuld':other.massa_niet_ingevuld,'straal_niet_ingevuld':other.straal_niet_ingevuld,'reus_proxy':other.reus_proxy,'zwaar_13MJ_of_meer':other.zwaar_13MJ_of_meer})
    pairs=pd.DataFrame(records)
    host_summary=pd.DataFrame(index=pd.Index(hosts,name='ster'))
    host_summary['groep1_planeten']=anchors.groupby('hostname').pl_name.apply(lambda x:'; '.join(x))
    for col,mask in [('andere_planeten_bekend',np.ones(len(pairs),dtype=bool)),('andere_reus_bekend',pairs.reus_proxy),('zware_begeleider_bekend',pairs.zwaar_13MJ_of_meer)]:
        host_summary[col]=host_summary.index.isin(pairs.loc[mask,'ster'])
    host_summary['aantal_bekende_planeten']=planets.groupby('hostname').size().reindex(host_summary.index,fill_value=0)
    host_summary['andere_reuzen']=pairs[pairs.reus_proxy].groupby('ster').begeleider.apply(lambda x:'; '.join(sorted(set(x)))).reindex(host_summary.index,fill_value='')
    host_summary['zware_begeleiders']=pairs[pairs.zwaar_13MJ_of_meer].groupby('ster').begeleider.apply(lambda x:'; '.join(sorted(set(x)))).reindex(host_summary.index,fill_value='')
    anchors.to_csv(OUT/'groep1_planeten.csv',index=False)
    planets.to_csv(OUT/'alle_bekende_planeten_bij_groep1_sterren.csv',index=False)
    pairs.to_csv(OUT/'alle_begeleider_paren.csv',index=False)
    host_summary.to_csv(OUT/'telling_per_ster.csv')
    pairs[pairs.reus_proxy].to_csv(OUT/'extra_reuzen.csv',index=False)
    sensitivity=[]
    for cutoff in [30,50,.3*MJ,100]:
        flag=((planets.bruikbare_massagrens&planets.pl_bmasse.ge(cutoff))|(planets.bruikbare_straalgrens&planets.pl_rade.ge(8)))&~planets.zwaar_13MJ_of_meer
        names=set(planets.loc[flag,'pl_name'])
        selected=pairs[pairs.begeleider.isin(names)]
        sensitivity.append({'massagrens_aarde':cutoff,'alternatieve_straalgrens_aarde':8,'sterren_met_extra_reus':int(selected.ster.nunique())})
    pd.DataFrame(sensitivity).to_csv(OUT/'gevoeligheid_massagrens.csv',index=False)
    n=len(host_summary);ng=int(host_summary.andere_reus_bekend.sum());nh=int(host_summary.zware_begeleider_bekend.sum());na=int(host_summary.andere_planeten_bekend.sum())
    # 13 MJ here is an operational split, not a definitive formation classification.
    result={'group1_planets':len(anchors),'host_stars':n,'hosts_with_other_known_planets':na,'hosts_with_extra_giant_proxy':ng,'fraction_hosts_with_extra_giant_proxy':ng/n,'hosts_with_heavy_companion':nh,'hosts_with_only_heavy_not_extra_giant':int((host_summary.zware_begeleider_bekend&~host_summary.andere_reus_bekend).sum()),'hosts_with_neither_extra_giant_nor_heavy_but_other_planets':int((host_summary.andere_planeten_bekend&~host_summary.andere_reus_bekend&~host_summary.zware_begeleider_bekend).sum()),'hosts_no_other_planets_known':int((~host_summary.andere_planeten_bekend).sum()),'distinct_extra_giant_companions':int(pairs.loc[pairs.reus_proxy,'begeleider'].nunique()),'giant_pairs':int(pairs.reus_proxy.sum()),'heavy_systems':host_summary.loc[host_summary.zware_begeleider_bekend,'zware_begeleiders'].to_dict(),'sensitivity':sensitivity,'definition':'At least 50 Earth masses from non-infilled mass or Msini (including lower bound), OR non-infilled radius >=8 Earth radii. Upper limits cannot qualify. Known mass/minimum mass >=13 MJ is reported separately. Composition not directly established.','source_sha256':hashlib.sha256((ROOT/'data/nasa_pscomppars.csv').read_bytes()).hexdigest(),'source_retrieved_utc':json.loads((ROOT/'data/provenance.json').read_text())['retrieved_utc'],'host_matching':'Exact NASA hostname; companions around a separately named binary star are not automatically merged.'}
    assert n==anchors.hostname.nunique()
    assert all(pairs.groep1_planeet.ne(pairs.begeleider))
    assert set(pairs.ster).issubset(set(hosts))
    # Check WASP-47 c: its recorded lower bound must not be discarded as an upper limit.
    assert pairs[pairs.begeleider.eq('WASP-47 c')].reus_proxy.all()
    assert ng+result['hosts_with_only_heavy_not_extra_giant']+result['hosts_with_neither_extra_giant_nor_heavy_but_other_planets']==na
    (OUT/'samenvatting.json').write_text(json.dumps(result,indent=2))
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    labels=['Andere gasreusachtige planeet bekend','Alleen zeer zware begeleider(s) bekend','Alleen kleinere andere planeten bekend','Geen andere planeet bekend']
    counts=[ng,result['hosts_with_only_heavy_not_extra_giant'],result['hosts_with_neither_extra_giant_nor_heavy_but_other_planets'],result['hosts_no_other_planets_known']]
    ax.barh(labels,counts,color=['#286c8e','#ae741c','#75865a','#b7bec3'])
    for i,v in enumerate(counts):ax.text(v+4,i,f'{v} ({v/n:.1%})',va='center')
    ax.set(xlabel='Aantal sterren met minstens één planeet uit groep 1',title=f'Bekende begeleiders bij {n} sterren\nNASA-momentopname 18 oktober 2026',xlim=(0,max(counts)*1.25))
    ax.invert_yaxis();fig.savefig(OUT/'begeleiders_overzicht.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))
    print(host_summary[host_summary.andere_planeten_bekend].to_string())
    return result

if __name__=='__main__':analyze()
