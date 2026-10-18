"""Reproduce the original online SOM rule using a current NASA snapshot.
Run with Python 3.12+, numpy, pandas, scikit-learn, matplotlib, numba.
"""
from pathlib import Path
import json, hashlib, platform
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_rand_score
from numba import njit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
FEATURES=['pl_bmasse','pl_rade','pl_dens','pl_insol','pl_eqt','pl_orbper','st_mass','st_rad','st_teff','st_lum']
LABELS=['Massa (aarde)','Straal (aarde)','Dichtheid (g/cm³)','Instraling (aarde)','Evenwichtstemp. (K)','Omlooptijd (dagen)','Stermassa (zon)','Sterstraal (zon)','Stertemperatuur (K)','Sterlichtkracht (zon)']
PARAMS=dict(nodes=12,epochs=1600,alpha=0.05,sigma=4.0)

@njit(cache=True)
def online_som(x,w,epochs=1600,alpha=0.05,sigma=4.0):
    # Same update as legacy TensorFlow code: exp(-distance²/sigma²).
    # Fixed row order and a complete pass per epoch, as in the old program.
    for epoch in range(epochs):
        decay=1.0-epoch/epochs
        rate=alpha*decay; radius=sigma*decay
        for row in range(x.shape[0]):
            winner=0; best=1e300
            for node in range(w.shape[0]):
                dist=0.0
                for f in range(x.shape[1]): dist+=(w[node,f]-x[row,f])**2
                if dist<best: best=dist;winner=node
            for node in range(w.shape[0]):
                influence=rate*np.exp(-((node-winner)**2)/(radius*radius))
                for f in range(x.shape[1]): w[node,f]+=influence*(x[row,f]-w[node,f])
    return w

def assign(x,w):
    distances=np.sum((x[:,None,:]-w[None,:,:])**2,axis=2)
    order=np.argsort(distances,axis=1)
    labels=order[:,0]
    return labels,float(np.sqrt(distances[np.arange(len(x)),labels]).mean()),float((np.abs(order[:,0]-order[:,1])>1).mean())

def prepare():
    raw=pd.read_csv(ROOT/'data/nasa_pscomppars.csv')
    assert raw.pl_name.is_unique
    meta=json.loads((ROOT/'data/provenance.json').read_text())
    assert hashlib.sha256((ROOT/'data/nasa_pscomppars.csv').read_bytes()).hexdigest()==meta['sha256']
    clean=raw.copy()
    # NASA st_lum is log10(L/Lsun), unlike the old linear luminosity column.
    clean['st_lum']=10.0**pd.to_numeric(clean.st_lum,errors='coerce')
    for c in FEATURES:
        clean[c]=pd.to_numeric(clean[c],errors='coerce')
        clean.loc[(~np.isfinite(clean[c])) | (clean[c]<=0) | (pd.to_numeric(clean[c+'lim'],errors='coerce').fillna(0)!=0),c]=np.nan
    eligible=(clean.pl_controv_flag.eq(0)) & clean[FEATURES].notna().all(axis=1)
    clean['mass_calculated']=clean.pl_bmasse_reflink.str.contains('calculated',case=False,na=False)
    clean['radius_calculated']=clean.pl_rade_reflink.str.contains('calculated',case=False,na=False)
    complete=clean.loc[eligible].copy().reset_index(drop=True)
    strict=complete.loc[~complete.mass_calculated & ~complete.radius_calculated & complete.pl_bmassprov.isin(['Mass','Msin(i)/sin(i)'])].copy().reset_index(drop=True)
    audit=pd.DataFrame({'eigenschap':LABELS,'NASA_kolom':FEATURES,'ontbrekend_of_limiet':clean[FEATURES].isna().sum().values,'berekend_door_NASA':[raw[c+'_reflink'].str.contains('calculated',case=False,na=False).sum() for c in FEATURES]})
    return raw,complete,strict,audit,meta

def fit(frame,seed,shuffle_order=False):
    scaler=MinMaxScaler();x=scaler.fit_transform(frame[FEATURES])
    w=np.random.default_rng(seed).uniform(0,1,(12,len(FEATURES)))
    training_x=x[np.random.default_rng(seed+100).permutation(len(x))] if shuffle_order else x
    w=online_som(training_x,w,**{k:v for k,v in PARAMS.items() if k!='nodes'})
    labels,qe,te=assign(x,w)
    return labels,w,x,scaler,{'seed':seed,'n':len(x),'quantization_error':qe,'topographic_error':te,'occupied_nodes':int(len(np.unique(labels)))}

def run():
    out=ROOT/'results';out.mkdir(exist_ok=True)
    raw,complete,strict,audit,meta=prepare()
    audit.to_csv(out/'datakwaliteit.csv',index=False)
    print(f'NASA: {len(raw)}; complete: {len(complete)}; strict: {len(strict)}',flush=True)
    labels,w,x,scaler,metrics=fit(complete,42)
    complete['SOM_groep']=labels+1
    complete.to_csv(out/'planeten_met_groepen.csv',index=False)
    np.savez(out/'som_model.npz',weights=w,minimum=scaler.data_min_,maximum=scaler.data_max_,features=np.array(FEATURES),seed=42)
    profiles=complete.groupby('SOM_groep')[FEATURES].median()
    profiles.insert(0,'aantal',complete.groupby('SOM_groep').size())
    profiles.to_csv(out/'groepsprofielen.csv')
    sizes=complete.SOM_groep.value_counts().reindex(range(1,13),fill_value=0).sort_index()
    metrics['sizes']={str(k):int(v) for k,v in sizes.items()}
    stability=[]
    for seed in [43,44]:
        alt,_,_,_,m=fit(complete,seed)
        m['adjusted_rand_vs_seed42']=float(adjusted_rand_score(labels,alt));stability.append(m)
        print('Repeat',m,flush=True)
    order_labels,_,_,_,order_metrics=fit(complete,42,shuffle_order=True)
    order_metrics['adjusted_rand_vs_main']=float(adjusted_rand_score(labels,order_labels))
    slabels,sw,sx,ss,sm=fit(strict,42)
    strict['SOM_groep']=slabels+1;strict.to_csv(out/'controle_ware_massa_en_straal.csv',index=False)
    shared=complete.set_index('pl_name').loc[strict.pl_name,'SOM_groep'].values
    sm['adjusted_rand_vs_main_shared_planets']=float(adjusted_rand_score(shared,slabels))
    strict.groupby('SOM_groep')[FEATURES].median().assign(aantal=strict.groupby('SOM_groep').size()).to_csv(out/'controle_groepsprofielen.csv')
    full_methods=raw.discoverymethod.value_counts(normalize=True)
    methods=pd.concat([full_methods.rename('archief'),complete.discoverymethod.value_counts(normalize=True).rename('hoofdproef'),strict.discoverymethod.value_counts(normalize=True).rename('controle')],axis=1).fillna(0)
    methods.to_csv(out/'selectie_ontdekkingsmethoden.csv')
    summary={'source':meta,'parameters':PARAMS,'feature_columns':FEATURES,'archive_rows':len(raw),'excluded':len(raw)-len(complete),'mass_calculated_in_main':int(complete.mass_calculated.sum()),'radius_calculated_in_main':int(complete.radius_calculated.sum()),'minimum_mass_in_main':int(complete.pl_bmassprov.eq('Msini').sum()),'main':metrics,'repeats':stability,'strict_control':sm,'fixed_shuffled_order_control':order_metrics,'python':platform.python_version()}
    (out/'samenvatting.json').write_text(json.dumps(summary,indent=2))
    plot(complete,strict,profiles,sizes,x,w,methods,out)
    print(json.dumps({k:v for k,v in summary.items() if k!='source'},indent=2),flush=True)
    return summary

def plot(data,strict,profiles,sizes,x,w,methods,out):
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.titlesize':15})
    fig,axes=plt.subplots(2,1,figsize=(12,9),gridspec_kw={'height_ratios':[1,2]},layout='constrained')
    axes[0].bar(sizes.index,sizes.values,color='#286c8e')
    for k,v in sizes.items():axes[0].text(k,v+max(sizes)*.015,str(v),ha='center',fontsize=10)
    axes[0].set(xticks=range(1,13),xlabel='SOM-plek (geen vooraf bekende planeetsoort)',ylabel='Aantal planeten',title=f'Bezetting van de twaalf SOM-plekken — {len(data):,} planeten',ylim=(0,max(sizes)*1.17))
    # Compare medians by within-sample percentile rank, not by raw physical units.
    ranks=data[FEATURES].rank(pct=True).groupby(data.SOM_groep).median().reindex(range(1,13))
    im=axes[1].imshow(ranks.T,aspect='auto',vmin=0,vmax=1,cmap='cividis')
    axes[1].set(xticks=np.arange(12),xticklabels=range(1,13),yticks=np.arange(10),yticklabels=LABELS,xlabel='SOM-plek',title='Kenmerken per plek: mediaan van de percentielrangen')
    fig.colorbar(im,ax=axes[1],label='Rang binnen gebruikte planeten (0 = laag, 1 = hoog)',shrink=.8)
    fig.savefig(out/'som_overzicht.png',dpi=160);plt.close(fig)
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.lines import Line2D
    palette=ListedColormap(plt.get_cmap('tab20')(np.arange(12)))
    norm=BoundaryNorm(np.arange(.5,13.5),12)
    fig,axes=plt.subplots(1,2,figsize=(13,6.3),layout='constrained')
    for ax,d,title in zip(axes,[data,strict],['Hoofdproef: inclusief ingevulde waarden','Controle: massa en straal niet ingevuld; ware massa']):
        sc=ax.scatter(d.pl_rade,d.pl_bmasse,c=d.SOM_groep,cmap=palette,norm=norm,s=12,alpha=.6,rasterized=True)
        ax.set(xscale='log',yscale='log',xlabel='Straal (aardstralen)',ylabel='Massa (aardmassa’s)',title=f'{title}\nn = {len(d):,}')
        ax.grid(alpha=.15)
    fig.suptitle('Massa en straal — groepskleuren horen bij afzonderlijke SOM-trainingen',fontsize=14)
    fig.legend(handles=[Line2D([],[],color=palette(i-1),marker='o',linestyle='',label=str(i)) for i in range(1,13)],loc='outside lower center',ncol=12,title='SOM-plek (nummers zijn per training bepaald)')
    fig.savefig(out/'massa_straal.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained')
    m=methods.loc[methods.max(axis=1).sort_values(ascending=False).index]
    m.mul(100).plot.barh(ax=ax,color=['#aeb7bd','#286c8e','#b77d22'])
    ax.set(xlabel='Aandeel (%)',ylabel='',title='Welke ontdekkingsmethoden blijven na selectie over?')
    ax.invert_yaxis();ax.legend(['Hele NASA-selectie','Hoofdproef','Controle'])
    fig.savefig(out/'selectiebias.png',dpi=160);plt.close(fig)

if __name__=='__main__':run()
