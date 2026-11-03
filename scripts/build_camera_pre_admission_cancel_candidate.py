"""Build non-deployed camera candidate for canceled requests before admission."""
from pathlib import Path
SOURCE=Path('integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua')
DEST=Path('integrations/ac_csp/candidates/FlyBrainCamera-pre-admission-cancel-v1.lua')
OLD="""          if staleTerminal then lastID = id else begin(request, now) end
"""
NEW="""          local safeID = type(id) == 'string' and #id <= 80
            and id:match('^[a-zA-Z0-9_-]+$')
          local directory = safeID and (root .. '/' .. id) or nil
          local canceledBeforeAdmission = safeID and request.expected_generation == generation
            and io.fileExists(directory .. '/stop') and not io.fileExists(directory .. '/result.json')
          if staleTerminal then
            lastID = id
          elseif canceledBeforeAdmission then
            -- Terminal precedes consuming ID: a failed write never hides the request.
            write(directory .. '/result.json', {status='stopped', frames=0,
              request_id=id, actuator_messages_sent=0, actor_stream_created=false,
              reason='owned stop observed before admission', camera_started=false})
            lastID = id
            terminalState, terminalReason = 'stopped', 'owned stop before admission'
            status('stopped', terminalReason)
          else
            begin(request, now)
          end
"""
def build():
    source=SOURCE.read_text(encoding='utf8')
    if source.count(OLD)!=1:raise ValueError('Exact source anchor required')
    return source.replace(OLD,NEW).encode('utf8')
if __name__=='__main__':
    DEST.write_bytes(build())
