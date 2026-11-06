"""Preserve original evidence and publish current offline failure status."""
import os
from datetime import datetime,timezone
from pathlib import Path
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from flybrain_drive.evidence import write_json
from scripts.control_ac_session import read_session_heartbeat

def main():
 root=Path('runs/paired-muscle-decoder-20261104');previous=read('runs/visual-turn-speed-20261104/source-receipt-final-01.json');old=read('runs/visual-turn-speed-20261104/summary-01.json');preserved=dict(previous['preserved_production_and_checkpoint_sha256']);sources=dict(previous['preserved_prior_experiment_source_sha256']);sources.update({p:h for p,h in previous['source_and_documents_sha256'].items() if p.startswith(('src/','scripts/','tests/'))})
 for arm in ['turn_only','turn_speed']:preserved[f'runs/visual-turn-speed-20261104/{arm}-fit-01/turn-speed.pt']=old['models'][arm]['checkpoint_sha256']
 assert all(sha256(p)==h for p,h in preserved.items());assert all(sha256(p)==h for p,h in sources.items())
 mono=read(root/'evaluation-monotone-01/result.json');free=read(root/'evaluation-free-01/result.json');reg=read(root/'registered-audit-01.json');summary=dict(status='completed_offline_paired_decoder_and_yaw_registration_trials',monotone=mono,free=free,registered=reg,yaw_audit_sha256=sha256(root/'yaw-audit-01.json'),promoted=False,actual_driving_improvement_verified=False,student_control_sent=False,simulator_mutation_api_calls=0,desktop_automation_calls=0,goal_status='active',next_work='Jointly preserve original command regression and correct physical yaw/speed semantics upstream; reserve new image seed before further learning.')
 write_json(root/'summary-01.json',summary)
 title='근육 판독 보정·좌우 명령 부호 감사';text='양의 power의 가속/감속 부호는 고쳤으나 paired 진입 전 조향 RMSE가 악화했다. 별도 affine 후보는 7구간 yaw 평균 방향을 바로잡았지만 gain 목표 실패·진입 전 조향 10.89~17.81% 악화·새 영상 방향 0/18이다. 14검사·native/camera128사건 parity·독립 NumPy 감사를 통과했으나 세 후보 미승격·학생 제어0·실제 주행/생리/온도 적응/한계 주행 미검증·목표active.'
 for name in ['README.md','docs/CONTEXT.md','docs/research-plan.md','docs/implementation-status.md']:
  path=Path(name);original=path.read_text(encoding='utf-8');first,rest=original.split('\n',1);link=('docs/' if name=='README.md' else '')+'paired-muscle-decoder-and-yaw-sign-20261104.md';path.write_text(first+'\n\n**2026-11-04 '+title+'：** '+text+' [原本・結果]('+link+').\n\n以下は以前の段階の当時の報告。\n'+rest,encoding='utf-8')
 # Korean status prose is explicit, and historical contents remain intact.
 for name in ['README.md','docs/CONTEXT.md','docs/research-plan.md','docs/implementation-status.md']:
  p=Path(name);t=p.read_text(encoding='utf-8').replace('[原本・結果]','[원본·결과]').replace('以下は以前の段階の当時の報告。','아래는 이전 단계의 당시 보고다.');p.write_text(t,encoding='utf-8')
 context=Path('docs/CONTEXT.md');t=context.read_text(encoding='utf-8');anchor='초파리의 겹눈/운동 시각';i=t.index(anchor);t=t[:i]+'최신 원본은 `runs/paired-muscle-decoder-20261104/summary-01.json`과 `source-receipt-final-01.json`이다. 모든 이번 프로세스는 종료됐으며 과거 turn/speed 단계와 구분한다. 다음은 판독기만 추가 선택하지 않고 기존 회귀와 물리 명령 의미를 함께 보존하는 상류 경계 공동 학습이다. 실제 조향·가감속·기울기·하중·타이어·변속·차종/셋업·상대풍 및 빠른 라인/랩타임은 계속 목표다.\n\n'+t[i:];context.write_text(t,encoding='utf-8')
 try:heartbeat=read_session_heartbeat(Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json')
 except (OSError,ValueError,RuntimeError) as e:heartbeat=dict(read_failed=True,error=type(e).__name__)
 files=list(Path('scripts').glob('*paired*decoder*.py'))+list(Path('scripts').glob('*registered*decoder*.py'))+[Path('scripts/collect_decoder_development_probes.py'),Path('scripts/collect_decoder_yaw_probes.py'),Path('src/flybrain_drive/paired_flight_vehicle_readout.py'),Path('src/flybrain_drive/paired_vehicle_candidate.py'),Path('tests/test_paired_flight_vehicle_readout.py')]+[Path(p) for p in ['README.md','docs/CONTEXT.md','docs/research-plan.md','docs/implementation-status.md','docs/paired-muscle-decoder-and-yaw-sign-20261104.md']]
 lua=Path('C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/apps/lua/FlyBrainCamera/FlyBrainCamera.lua');assert sha256(lua)=='0c20d720a7a2bf20a7e30fdc7370ee5e9932babf71fb360a78a5f07719360b7e'
 write_json(root/'source-receipt-final-01.json',dict(observed_utc=datetime.now(timezone.utc).isoformat(),source_revision=previous['source_revision'],preserved_production_and_checkpoint_sha256=preserved,preserved_prior_experiment_source_sha256=sources,source_and_documents_sha256={str(p).replace('\\','/'):sha256(p) for p in files},summary_sha256=sha256(root/'summary-01.json'),heartbeat_readonly=heartbeat,installed_lua_sha256=sha256(lua),process_inventory_sha256=sha256(root/'process-inventory-final-01.json'),observed_original_terminal_handles_exit0=[70129,81897,37169,43296,38891,78116,13539,65008,59722,55258],tests_passed=14,actor_promoted=False,student_control_sent=False,simulator_mutation_api_calls=0,desktop_automation_calls=0,goal_status='active'))
 print(dict(preserved_files=len(preserved),prior_sources=len(sources),status=summary['status']))
if __name__=='__main__':main()
