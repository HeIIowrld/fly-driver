import json
import pytest
from scripts.sac_cohort_contract import three_episode_cohort


@pytest.mark.parametrize('leak',['third_train','duplicate_episode','duplicate_directory'])
def test_native_normalization_cohort_rejects_role_or_episode_leakage(tmp_path,leak):
    entries=[dict(role=role,episode_id=str(i),data=str(tmp_path/str(i)))
             for i,role in enumerate(['train','train','development'])]
    if leak=='third_train':entries[2]['role']='train'
    if leak=='duplicate_episode':entries[2]['episode_id']='0'
    if leak=='duplicate_directory':entries[2]['data']=entries[0]['data']
    p=tmp_path/'split.json';p.write_text(json.dumps(dict(episodes=entries)),encoding='utf-8')
    with pytest.raises(ValueError):three_episode_cohort(p)
