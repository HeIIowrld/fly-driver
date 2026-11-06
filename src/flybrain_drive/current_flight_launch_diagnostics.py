"""Original420-byte native gear/engine/clutch diagnostic; excluded from actor inputs."""
import math,struct
from .ac_teacher_physics import decode_teacher_physics
from .ac_native_physics import PHYSICS_PREFIX,PHYSICS_LAYOUT
CLUTCH_OFFSET=364

def decode_launch_diagnostics(raw,received):
 sample=decode_teacher_physics(raw,received);values=PHYSICS_PREFIX.unpack(raw[:PHYSICS_PREFIX.size]);offset=0;result={}
 for name,_,width in PHYSICS_LAYOUT:
  if name in ['gear','rpms','autoShifterOn','pitLimiterOn']:result[name]=values[offset]
  offset+=width
 clutch=struct.unpack_from('<f',raw,CLUTCH_OFFSET)[0]
 if not math.isfinite(clutch) or not 0<=clutch<=1:raise ValueError('Native diagnostic clutch outside fraction0..1')
 result.update(clutch=clutch,actor_input=False,packet=sample.physics.packet_id,builtin_ai_controlled=sample.builtin_ai_controlled)
 return result
