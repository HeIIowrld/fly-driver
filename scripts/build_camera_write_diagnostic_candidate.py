"""Build a non-deployed diagnostic-only Lua candidate from an exact source."""
from pathlib import Path
import hashlib

SOURCE=Path('integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua')
TARGET=Path('integrations/ac_csp/candidates/FlyBrainCamera-write-path-diagnostic.lua')
EXPECTED='475e085d974595aa503fa687d58913cf1551b39fb4d230721763f5fd68a8d119'
OLD="assert(saved, 'Cannot commit camera metadata')"
NEW="assert(saved, 'Cannot commit camera metadata: ' .. tostring(path))"

def build():
 raw=SOURCE.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=EXPECTED:raise ValueError('Exact observed deployed source required')
 text=raw.decode('utf8')
 if text.count(OLD)!=1:raise ValueError('Single original write boundary required')
 return text.replace(OLD,NEW,1).encode('utf8')

if __name__=='__main__':
 data=build();TARGET.parent.mkdir(parents=True,exist_ok=True)
 if TARGET.exists() and TARGET.read_bytes()!=data:raise FileExistsError('Different candidate exists')
 TARGET.write_bytes(data)
 print('candidate',TARGET,'sha256',hashlib.sha256(data).hexdigest(),'not deployed')
