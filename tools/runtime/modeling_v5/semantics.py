"""Generic model-side checks, never ground-truth labels or problem-family rules."""
from .sources import catalog


def diagnostics(model, question, bindings):
    n = len(model['variables'])
    sources = [r for r in catalog(question) if r['kind']=='vector' and len(r['value'])==n]
    vectors = [('objective.linear',model['objective']['linear'])]+[('constraint:'+c['id'],c['a']) for c in model['constraints']]
    bound = {r['target'] for r in bindings}
    mismatches = []
    for target,a in vectors:
        if target in bound or not sources or n < 4:
            continue
        nearest = max(sources,key=lambda r:sum(x==y for x,y in zip(a,r['value'])))
        indices = [i for i,(x,y) in enumerate(zip(a,nearest['value'])) if x!=y]
        if indices and len(indices) <= .4*n:
            mismatches.append({'target':target,'source_candidate':nearest['id'],
                'different_entries':len(indices),
                'examples':[{'index':i,'model':a[i],'source':nearest['value'][i]} for i in indices[:8]],
                'action':'Check whether this source array is intended. If so use a source reference; otherwise explain the transformation. Not an automatic correction.'})
    forced = []
    for c in model['constraints']:
        nz = [(i,v) for i,v in enumerate(c['a']) if v]
        if c['op']=='eq' and len(nz)==1:
            i,a = nz[0]
            forced.append({'constraint':c['id'],'variable':model['variables'][i]['name'],
                           'coefficient':a,'rhs':c['rhs'],
                           'check':'This equality fixes one variable. Verify that the source really requires a fixed value, rather than only domain membership or exclusivity.'})
    probes = []
    for i,j,c in model['objective']['quadratic'][:6]:
        if i==j or any((model['variables'][k]['lower'],model['variables'][k]['upper'])!=(0,1) for k in (i,j)):
            continue
        # Exact restricted polynomial, other variables held at zero; this is not a feasible-solution oracle.
        li = model['objective']['linear'][i]+sum(v for a,b,v in model['objective']['quadratic'] if a==b==i)
        lj = model['objective']['linear'][j]+sum(v for a,b,v in model['objective']['quadratic'] if a==b==j)
        probes.append({'indices':[i,j],'others_fixed_to_zero':True,
                       'values_00_01_10_11':[0,lj,li,li+lj+c],
                       'check':'Compare these exact objective values to the intended relationship in the text. These probes may violate other constraints.'})
    return {'source_bindings':bindings,'suspected_numeric_copy_errors':mismatches[:12],
            'single_variable_equalities':forced[:12],'single_variable_equalities_total':len(forced),
            'objective_local_probes':probes,
            'scope':'Source literals and predicted mathematics only; hints may be false positives, not proof of semantics.'}
