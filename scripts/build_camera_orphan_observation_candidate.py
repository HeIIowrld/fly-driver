"""Build non-deployed candidate distinguishing a prior-generation orphan.

An orphan observation is not a terminal receipt and does not authorize another
camera request. The host must independently establish old-process termination.
"""
from pathlib import Path
from scripts.build_camera_pre_admission_cancel_candidate import SOURCE, OLD, NEW

DEST = Path('integrations/ac_csp/candidates/FlyBrainCamera-pre-admission-cancel-v2.lua')
ANCHOR = '          elseif canceledBeforeAdmission then\n'
BRANCH = '''          elseif safeID and type(request.expected_generation) == 'string'
            and request.expected_generation ~= '' and request.expected_generation ~= generation
            and io.fileExists(directory .. '/stop') and not io.fileExists(directory .. '/result.json') then
            -- This producer cannot know what the prior producer did. Never forge
            -- its terminal receipt or imply the old process has terminated.
            local observation = directory .. '/orphan-observation.json'
            assert(not io.fileExists(observation), 'Prior orphan observation requires host review')
            write(observation, {schema='flybrain.csp.camera.orphan_observation.v1',
              status='prior_generation_unresolved', request_id=id,
              expected_generation=request.expected_generation, observer_generation=generation,
              original_terminal_present=false, stop_present=true, old_process_exit_verified=false,
              prior_camera_started='unknown', prior_frames='unknown', actor_admissible=false})
            lastID = id
            terminalState, terminalReason = 'failed', 'prior generation unresolved; host review required'
            status('failed', terminalReason)
'''

def build():
    source = SOURCE.read_text(encoding='utf8')
    if source.count(OLD) != 1 or NEW.count(ANCHOR) != 1:
        raise ValueError('Exact original and cancellation anchors required')
    return source.replace(OLD, NEW.replace(ANCHOR, BRANCH + ANCHOR)).encode('utf8')

if __name__ == '__main__':
    DEST.write_bytes(build())
