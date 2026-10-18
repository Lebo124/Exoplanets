"""One controlled experiment: log10 then min-max, same 12-node online SOM."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_rand_score
from analysis import prepare, online_som, assign, FEATURES, LABELS, PARAMS, ROOT
OUT=ROOT/'results_log'

def solar_system():
    # Same NASA fact-sheet inputs and Teq assumption as the previous projection.
    names=['Mercurius','Venus','Aarde','Mars','Jupiter','Saturnus','Uranus','Neptunus']
    a=np.array([57.9,108.2,149.6,228,778.5,1432,2867,4515])/149.6
    data=np.column_stack([
        np.array([.330,4.87,5.97,.642,1898,568,86.8,102])/5.97,
        np.array([4879,12104,12756,6792,142984,120536,51118,49528])/12756,
        np.array([5429,5243,5514,3934,1326,687,1270,1638])/1000,
        a**-2,5772*np.sqrt(.00465046726/(2*a)),
        np.array([88,224.7,365.2,687,4331,10747,30589,59800]),
        np.ones(8),np.ones(8),np.full(8,5772),np.ones(8)])
    return pd.DataFrame(data,index=pd.Index(names,name='planeet'),columns=FEATURES)

def run():
    OUT.mkdir(exist_ok=True)
    raw,data,strict,audit,provenance=prepare()
    old=pd.read_csv(ROOT/'results/planeten_met_groepen.csv')
    assert np.array_equal(old.pl_name.values,data.pl_name.values)
    # Exactly the same rows, order, initialization and update; only preprocessing changes.
    np.testing.assert_allclose(old[FEATURES],data[FEATURES],rtol=1e-12,atol=1e-12)
    values=data[FEATURES].to_numpy()
    assert np.isfinite(values).all() and (values>0).all()
    scaler=MinMaxScaler(); x=scaler.fit_transform(np.log10(values))
    weights=np.random.default_rng(42).uniform(0,1,(12,len(FEATURES)))
    weights=online_som(x,weights,epochs=PARAMS['epochs'],alpha=PARAMS['alpha'],sigma=PARAMS['sigma'])
    labels,qe,te=assign(x,weights)
    data['SOM_groep']=labels+1
    data.to_csv(OUT/'planeten_met_groepen_log.csv',index=False)
    np.savez(OUT/'som_model_log.npz',weights=weights,log_minimum=scaler.data_min_,log_maximum=scaler.data_max_,features=np.array(FEATURES),seed=42)
    profiles=data.groupby('SOM_groep')[FEATURES].median()
    profiles.insert(0,'aantal',data.groupby('SOM_groep').size())
    profiles.to_csv(OUT/'groepsprofielen_log.csv')
    cross=pd.crosstab(old.SOM_groep,data.SOM_groep).reindex(index=range(1,13),columns=range(1,13),fill_value=0)
    cross.index.name='oude_groep';cross.columns.name='log_groep';cross.to_csv(OUT/'oude_naar_log_groepen.csv')
    solar=solar_system(); sx=scaler.transform(np.log10(solar.values))
    sd=np.linalg.norm(sx[:,None,:]-weights[None,:,:],axis=2); order=np.argsort(sd,axis=1)
    om=np.load(ROOT/'results/som_model.npz');oz=(solar.values-om['minimum'])/(om['maximum']-om['minimum'])
    olddist=np.linalg.norm(oz[:,None,:]-om['weights'][None,:,:],axis=2)
    solar['oude_groep']=np.argmin(olddist,axis=1)+1
    solar['log_groep']=order[:,0]+1
    solar['tweede_log_groep']=order[:,1]+1
    solar['afstand_log']=sd[np.arange(8),order[:,0]]
    solar['tweede_afstand_log']=sd[np.arange(8),order[:,1]]
    solar.to_csv(OUT/'zonnestelsel_log.csv')
    sizes=data.SOM_groep.value_counts().reindex(range(1,13),fill_value=0).sort_index()
    result={'n':len(data),'parameters':PARAMS,'seed':42,'transform':'log10 of all 10 positive physical feature values, then per-feature MinMaxScaler; no clipping','source_sha256':provenance['sha256'],'source_retrieved_utc':provenance['retrieved_utc'],'old_model_sha256':hashlib.sha256((ROOT/'results/som_model.npz').read_bytes()).hexdigest(),'quantization_error_log_space':qe,'topographic_error':te,'adjusted_rand_vs_linear':float(adjusted_rand_score(old.SOM_groep,labels)),'occupied_groups':int((sizes>0).sum()),'group_sizes':{str(k):int(v) for k,v in sizes.items()},'solar_sources':['https://nssdc.gsfc.nasa.gov/planetary/factsheet/','https://exoplanetarchive.ipac.caltech.edu/docs/pscp_calc.html'],'solar_out_of_range_features':{name:[FEATURES[j] for j in range(10) if sx[i,j]<0 or sx[i,j]>1] for i,name in enumerate(solar.index)}}
    assert sizes.sum()==len(data)
    reconstructed=assign(scaler.transform(np.log10(data[FEATURES].values)),weights)[0]+1
    assert np.array_equal(reconstructed,data.SOM_groep.values)
    (OUT/'samenvatting_log.json').write_text(json.dumps(result,indent=2))
    plot(data,sizes,solar)
    print(json.dumps(result,indent=2))
    print(solar[['oude_groep','log_groep','tweede_log_groep','afstand_log']].to_string())
    return result

def plot(data,sizes,solar):
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    fig,axes=plt.subplots(2,1,figsize=(12,9),gridspec_kw={'height_ratios':[1,2]},layout='constrained')
    axes[0].bar(sizes.index,sizes.values,color='#286c8e')
    for k,v in sizes.items():axes[0].text(k,v+max(sizes)*.012,str(v),ha='center')
    axes[0].set(xticks=range(1,13),xlabel='Groep in de SOM met logaritmen',ylabel='Aantal planeten',title=f'Logaritmische schaling — {len(data):,} planeten, 12 groepen',ylim=(0,max(sizes)*1.14))
    rank=data[FEATURES].rank(pct=True).groupby(data.SOM_groep).median().reindex(range(1,13))
    im=axes[1].imshow(rank.T,aspect='auto',vmin=0,vmax=1,cmap='cividis')
    axes[1].set(xticks=np.arange(12),xticklabels=range(1,13),yticks=np.arange(10),yticklabels=LABELS,xlabel='Groep',title='Groepskenmerken: mediaan van percentielrangen')
    fig.colorbar(im,ax=axes[1],label='Laag (0) tot hoog (1) binnen de gebruikte planeten',shrink=.8)
    fig.savefig(OUT/'som_log_overzicht.png',dpi=160);plt.close(fig)

if __name__=='__main__':run()
