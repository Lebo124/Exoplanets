"""PCA95 on the existing log/min-max data, followed by the same 12-node SOM."""
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from analysis import prepare,online_som,assign,FEATURES,LABELS,PARAMS,ROOT
from log_som import solar_system
OUT=ROOT/'results_pca'
THRESHOLD=.95

def run():
    OUT.mkdir(exist_ok=True)
    _,data,_,_,provenance=prepare()
    old=pd.read_csv(ROOT/'results_log/planeten_met_groepen_log.csv')
    assert np.array_equal(data.pl_name.values,old.pl_name.values)
    np.testing.assert_allclose(data[FEATURES],old[FEATURES],rtol=1e-12,atol=1e-12)
    base=np.load(ROOT/'results_log/som_model_log.npz')
    x=(np.log10(data[FEATURES].values)-base['log_minimum'])/(base['log_maximum']-base['log_minimum'])
    pca=PCA(n_components=None,svd_solver='full',whiten=False).fit(x)
    all_scores=pca.transform(x)
    ratios=pca.explained_variance_ratio_; cumulative=np.cumsum(ratios)
    k=int(np.searchsorted(cumulative,THRESHOLD)+1)
    scores=all_scores[:,:k]
    assert k<=10 and cumulative[k-1]>=THRESHOLD
    cov=np.cov(scores,rowvar=False)
    max_cov=float(np.abs(cov-np.diag(np.diag(cov))).max())
    assert max_cov<1e-10
    # Full PCA is an orthonormal rotation: pairwise distances and old assignments agree.
    check=x[:40,None,:]-x[None,:40,:]
    rotated=all_scores[:40,None,:]-all_scores[None,:40,:]
    np.testing.assert_allclose((check**2).sum(axis=2),(rotated**2).sum(axis=2),atol=1e-12)
    assert np.array_equal(assign(all_scores,pca.transform(base['weights']))[0]+1,old.SOM_groep.values)
    init=np.random.default_rng(42).uniform(0,1,(12,10))
    # Project the same original start weights, rather than choosing a new initialization.
    initial=pca.transform(init)[:,:k].copy()
    weights=online_som(scores,initial,epochs=PARAMS['epochs'],alpha=PARAMS['alpha'],sigma=PARAMS['sigma'])
    raw_labels=assign(scores,weights)[0]
    reverse_numbering=bool(np.sum((raw_labels+1-old.SOM_groep.values)**2)>np.sum((12-raw_labels-old.SOM_groep.values)**2))
    if reverse_numbering: weights=weights[::-1].copy()
    labels,qe,te=assign(scores,weights)
    data['SOM_groep']=labels+1
    data.to_csv(OUT/'planeten_met_groepen_pca.csv',index=False)
    pc_names=[f'PC{i+1}' for i in range(10)]
    variance=pd.DataFrame({'component':pc_names,'verklaarde_spreiding':ratios,'cumulatief':cumulative,'bewaard':[i<k for i in range(10)]})
    variance.to_csv(OUT/'pca_spreiding.csv',index=False)
    loadings=pd.DataFrame(pca.components_.T,index=FEATURES,columns=pc_names)
    loadings.index.name='NASA_eigenschap';loadings.to_csv(OUT/'pca_coefficienten.csv')
    pd.DataFrame(scores,columns=pc_names[:k]).assign(pl_name=data.pl_name.values,SOM_groep=labels+1).to_csv(OUT/'pca_scores.csv',index=False)
    profile=data.groupby('SOM_groep')[FEATURES].median()
    profile.insert(0,'aantal',data.groupby('SOM_groep').size());profile.to_csv(OUT/'groepsprofielen_pca.csv')
    cross=pd.crosstab(old.SOM_groep,data.SOM_groep).reindex(index=range(1,13),columns=range(1,13),fill_value=0)
    cross.index.name='log_groep';cross.columns.name='pca_groep';cross.to_csv(OUT/'log_naar_pca_groepen.csv')
    np.savez(OUT/'som_model_pca.npz',weights=weights,pca_mean=pca.mean_,pca_components=pca.components_,pca_explained_variance=pca.explained_variance_,pca_variance_ratio=ratios,retained_components=k,log_minimum=base['log_minimum'],log_maximum=base['log_maximum'],features=np.array(FEATURES),seed=42)
    solar=solar_system();sx=(np.log10(solar.values)-base['log_minimum'])/(base['log_maximum']-base['log_minimum'])
    solar_scores=pca.transform(sx)[:,:k]
    dist=np.linalg.norm(solar_scores[:,None,:]-weights[None,:,:],axis=2);order=np.argsort(dist,axis=1)
    solar['log_groep']=assign(sx,base['weights'])[0]+1
    solar['pca_groep']=order[:,0]+1;solar['tweede_pca_groep']=order[:,1]+1
    solar['afstand_pca']=dist[np.arange(8),order[:,0]];solar['tweede_afstand_pca']=dist[np.arange(8),order[:,1]]
    solar['verlies_door_projectie']=np.linalg.norm(sx-pca.inverse_transform(np.pad(solar_scores,((0,0),(0,10-k)))),axis=1)
    solar.to_csv(OUT/'zonnestelsel_pca.csv')
    train_dist=np.linalg.norm(scores-weights[labels],axis=1)
    solar['percentiel_afstand_binnen_trainingsset']=[float((train_dist<=d).mean()) for d in solar.afstand_pca]
    solar.to_csv(OUT/'zonnestelsel_pca.csv')
    sizes=data.SOM_groep.value_counts().reindex(range(1,13),fill_value=0).sort_index()
    result={'n':len(data),'retained_components':k,'variance_threshold':THRESHOLD,'retained_variance':float(cumulative[k-1]),'all_variance_ratios':ratios.tolist(),'whiten':False,'map_reversed_for_reading_direction':reverse_numbering,'parameters':PARAMS,'seed':42,'initialization':'Same 12x10 log-model starting weights projected into retained PCA space','max_off_diagonal_score_covariance':max_cov,'full_PCA_preserves_distances_and_old_assignments':True,'quantization_error_PCA_space':qe,'topographic_error':te,'adjusted_rand_vs_log':float(adjusted_rand_score(old.SOM_groep,labels)),'occupied_groups':int((sizes>0).sum()),'group_sizes':{str(a):int(b) for a,b in sizes.items()},'source_sha256':provenance['sha256'],'source_retrieved_utc':provenance['retrieved_utc'],'baseline_log_model_sha256':hashlib.sha256((ROOT/'results_log/som_model_log.npz').read_bytes()).hexdigest(),'training_residual_distance_quantiles':{str(q):float(np.quantile(train_dist,q)) for q in [.5,.9,.95,.99]}}
    assert sizes.sum()==len(data)
    np.testing.assert_array_equal(assign(scores,weights)[0]+1,data.SOM_groep.values)
    (OUT/'samenvatting_pca.json').write_text(json.dumps(result,indent=2))
    plot(data,sizes,variance,loadings,k)
    print(json.dumps(result,indent=2))
    print(loadings.iloc[:,:k].round(3).to_string())
    print(solar[['log_groep','pca_groep','tweede_pca_groep','afstand_pca','percentiel_afstand_binnen_trainingsset']].round(4).to_string())
    return result

def plot(data,sizes,variance,loadings,k):
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    fig,axes=plt.subplots(2,1,figsize=(11,9),layout='constrained')
    colors=['#286c8e' if i<k else '#b7bec3' for i in range(10)]
    axes[0].bar(np.arange(1,11),variance.verklaarde_spreiding*100,color=colors)
    axes[0].plot(np.arange(1,11),variance.cumulatief*100,'o-',color='#ae741c',label='Samen behouden')
    axes[0].axhline(95,color='#777777',linestyle='--',linewidth=1,label='Grens 95%')
    axes[0].set(xticks=range(1,11),xlabel='Hoofdcomponent',ylabel='Aandeel spreiding (%)',title=f'PCA: van tien eigenschappen naar {k} hoofdcomponenten',ylim=(0,105))
    axes[0].legend()
    limit=max(.01,abs(loadings.iloc[:,:k]).values.max())
    im=axes[1].imshow(loadings.iloc[:,:k].values,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
    axes[1].set(xticks=np.arange(k),xticklabels=[f'PC{i+1}' for i in range(k)],yticks=np.arange(10),yticklabels=LABELS,title='Welke eigenschappen vormen samen een component?')
    for i in range(10):
        for j in range(k):
            val=loadings.iloc[i,j];axes[1].text(j,i,f'{val:.2f}',ha='center',va='center',fontsize=10,color='white' if abs(val)>.7*limit else '#111111')
    fig.colorbar(im,ax=axes[1],label='Coëfficiënt in log/min-maxruimte',shrink=.8)
    fig.savefig(OUT/'pca_uitleg.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,9),gridspec_kw={'height_ratios':[1,2]},layout='constrained')
    axes[0].bar(sizes.index,sizes.values,color='#286c8e')
    for g,n in sizes.items():axes[0].text(g,n+max(sizes)*.012,str(n),ha='center')
    axes[0].set(xticks=range(1,13),xlabel='PCA-SOM-groep',ylabel='Aantal planeten',title=f'Logaritmen + PCA — {len(data):,} planeten, twaalf groepen',ylim=(0,max(sizes)*1.15))
    ranks=data[FEATURES].rank(pct=True).groupby(data.SOM_groep).median().reindex(range(1,13))
    im=axes[1].imshow(ranks.T,aspect='auto',vmin=0,vmax=1,cmap='cividis')
    axes[1].set(xticks=np.arange(12),xticklabels=range(1,13),yticks=np.arange(10),yticklabels=LABELS,xlabel='PCA-SOM-groep',title='Groepskenmerken in de oorspronkelijke eigenschappen')
    fig.colorbar(im,ax=axes[1],label='Mediaan van percentielrangen: laag (0) tot hoog (1)',shrink=.8)
    fig.savefig(OUT/'som_pca_overzicht.png',dpi=160);plt.close(fig)

if __name__=='__main__':run()
