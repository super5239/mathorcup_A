"""Numerical checks on explicitly supplied mg/L coordinates, not sensory certification.
AI assistance: OpenAI Codex, 2026-10-25.
"""
from pathlib import Path
import math,json

CONFIG_PATH=Path(__file__).with_name('defect_thresholds.json')

def check_transfer(source_mg_L,candidate_mg_L,config=None):
    """Never turns missing chemical/sensory information into a passed guarantee."""
    config=config or json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    items={}
    for name,spec in config['measured_defects'].items():
        a=source_mg_L.get(name);b=candidate_mg_L.get(name);threshold=spec['threshold_mg_L']
        valid=(isinstance(a,(int,float)) and isinstance(b,(int,float)) and
               math.isfinite(a) and math.isfinite(b) and a>=0 and b>=0)
        if not valid:
            items[name]={'status':'not_evaluable','reason':'missing_nonfinite_or_negative_concentration'}
            continue
        es=max(a/threshold-1,0);ec=max(b/threshold-1,0)
        items[name]={'status':'evaluated_under_declared_units','source_mg_L':a,'candidate_mg_L':b,
                     'threshold_mg_L':threshold,'source_excess':es,'candidate_excess':ec,
                     'concentration_nonincrease':bool(b<=a),
                     'excess_nonworsening':bool(ec<=es),
                     'candidate_below_threshold':bool(b<=threshold)}
    complete=all(v['status']=='evaluated_under_declared_units' for v in items.values())
    return {'items':items,
            'all_four_evaluable':complete,
            'all_four_concentration_nonincrease':complete and all(v['concentration_nonincrease'] for v in items.values()),
            'all_four_excess_nonworsening':complete and all(v['excess_nonworsening'] for v in items.values()),
            'all_four_below_threshold':complete and all(v['candidate_below_threshold'] for v in items.values()),
            'Stale_hops':{'status':'not_chemically_verifiable','reason':'isovaleric_acid_not_measured; ethyl_isovalerate_is_not_isovaleric_acid'},
            'all_five_chemically_certified':False,
            'qualification':'Numerical feasibility under supplied coordinates and thresholds; not a physical or sensory guarantee.'}

if __name__=='__main__':
    base={'DMS':.01,'Diacetyl':.05,'4VG':.1,'t2n':.00005}
    assert check_transfer(base,base)['all_four_concentration_nonincrease']
    small={k:v/2 for k,v in base.items()}
    assert check_transfer(base,small)['all_four_concentration_nonincrease']
    under=base.copy();under['DMS']=.015
    got=check_transfer(base,under)
    assert not got['all_four_concentration_nonincrease'] and got['all_four_excess_nonworsening']
    high=base.copy();high['DMS']=.03
    lower=high.copy();lower['DMS']=.025
    got=check_transfer(high,lower)
    assert got['all_four_concentration_nonincrease'] and not got['all_four_below_threshold']
    for invalid in [None,float('nan'),-1]:
        changed=base.copy();changed['t2n']=invalid
        got=check_transfer(base,changed)
        assert not got['all_four_evaluable'] and not got['all_four_excess_nonworsening']
    assert not check_transfer(base,base)['all_five_chemically_certified']
    tiny=base.copy();tiny['DMS']=math.nextafter(base['DMS'],math.inf)
    assert not check_transfer(base,tiny)['all_four_concentration_nonincrease']
    print('Constraint checker passed: unchanged/decreased values, below-threshold increases, existing exceedance, missing/nonfinite/negative values, and unmeasured Stale hops.')
