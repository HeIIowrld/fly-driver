"""Generate NEW-only explicit MMF status-policy derivatives; no deployment/live calls."""
# ruff: noqa: E501 -- Exact frozen source fragments retain their reviewable context.
import argparse
import hashlib
import json
import re
from pathlib import Path

BASES = {
 'integrations/ac_csp/candidates/FlyBrainCamera-heartbeat-wire-v2.lua': '807e708e098654cfaf8cf58ebd91efc15f850d3da31795b02d28bc063eeb024b',
}

LUA = '''-- BEGIN explicit MMF camera status lifecycle policy v1.
local CAMERA_STATUS_POLICY = 'mmf_lifecycle_only_v1'
local function acceptCameraStatusPolicy(request)
  if request.camera_status_policy == nil then return nil end
  assert(type(request.camera_status_policy) == 'string'
    and request.camera_status_policy == CAMERA_STATUS_POLICY, 'Unknown camera status policy')
  assert(request.mmf_metadata == true and request.shared_transport == true,
    'Lifecycle-only status requires explicit shared MMF transport')
  return CAMERA_STATUS_POLICY
end
local function cameraStatusPolicyEcho(owner)
  if owner and owner.camera_status_policy == CAMERA_STATUS_POLICY then
    return {schema='flybrain.csp.camera.status_policy_echo.v1', policy=CAMERA_STATUS_POLICY,
      request_id=owner.request.id, generation=generation}
  end
  return nil
end
local function writeCameraTerminal(owner, path, value)
  if owner.camera_status_policy == CAMERA_STATUS_POLICY then
    value.camera_status_policy_echo = cameraStatusPolicyEcho(owner)
  end
  write(path, value)
end
-- END explicit MMF camera status lifecycle policy v1.
'''


def checked(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASES[path]:
        raise ValueError('Frozen parent source changed: '+path)
    return raw.decode().replace('\r\n', '\n')


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Exact unique source edit required: '+old[:90])
    return text.replace(old, new)


def lua_source():
    text = checked('integrations/ac_csp/candidates/FlyBrainCamera-heartbeat-wire-v2.lua')
    text = once(text, 'local function begin(request, now)\n', LUA+
        'local function begin(request, now)\n  local acceptedStatusPolicy = acceptCameraStatusPolicy(request)\n')
    text = once(text, 'active = {directory = directory, request = request, frames = 0, started_clock = now,',
        'active = {directory = directory, request = request, camera_status_policy = acceptedStatusPolicy,\n    frames = 0, started_clock = now,')
    text = once(text, "  if active.mmfFrame then\n    metadata.metadata_transport = 'mmf_slot_sha256_v1'",
        "  if active.camera_status_policy == CAMERA_STATUS_POLICY then\n"
        "    metadata.camera_status_policy_echo = cameraStatusPolicyEcho(active)\n"
        "  end\n  if active.mmfFrame then\n    metadata.metadata_transport = 'mmf_slot_sha256_v1'")
    text, count = re.subn(r"write\((active|owner)\.directory \.\. '/result.json',",
        lambda m: "writeCameraTerminal("+m[1]+", "+m[1]+".directory .. '/result.json',", text)
    if count != 5:
        raise ValueError('All five camera terminal write callsites required')
    text = once(text, "      status(active and 'capturing' or terminalState, terminalReason)\n",
        "      if not (active and active.camera_status_policy == CAMERA_STATUS_POLICY) then\n"
        "        status(active and 'capturing' or terminalState, terminalReason)\n      end\n")
    return text


def inverse_lua(text):
    text = once(text, LUA+'local function begin(request, now)\n'
        '  local acceptedStatusPolicy = acceptCameraStatusPolicy(request)\n',
        'local function begin(request, now)\n')
    text = once(text,
        'active = {directory = directory, request = request, camera_status_policy = acceptedStatusPolicy,\n    frames = 0, started_clock = now,',
        'active = {directory = directory, request = request, frames = 0, started_clock = now,')
    text = once(text,
        "  if active.camera_status_policy == CAMERA_STATUS_POLICY then\n"
        "    metadata.camera_status_policy_echo = cameraStatusPolicyEcho(active)\n"
        "  end\n  if active.mmfFrame then\n    metadata.metadata_transport = 'mmf_slot_sha256_v1'",
        "  if active.mmfFrame then\n    metadata.metadata_transport = 'mmf_slot_sha256_v1'")
    text = once(text,
        "      if not (active and active.camera_status_policy == CAMERA_STATUS_POLICY) then\n"
        "        status(active and 'capturing' or terminalState, terminalReason)\n      end\n",
        "      status(active and 'capturing' or terminalState, terminalReason)\n")
    text, count = re.subn(r"writeCameraTerminal\((active|owner), \1\.directory \.\. '/result.json',",
        lambda m: "write("+m[1]+".directory .. '/result.json',", text)
    if count != 5:
        raise ValueError('Five exact terminal callsites required by inverse')
    # checked() normalizes source text; the inverse restores the parent's exact
    # original line ending as well as every explicit semantic edit.
    original = Path(next(iter(BASES))).read_bytes()
    if b'\r\n' in original:
        text = text.replace('\n', '\r\n')
    return text


def build():
    from flybrain_drive.camera_status_policy_v1 import profile_record

    return {
        Path('integrations/ac_csp/candidates/FlyBrainCamera-mmf-lifecycle-status-v1.lua'):
            lua_source().encode(),
        Path('configs/ac-mmf-lifecycle-status-v1-20261030.json'):
            (json.dumps(profile_record(), indent=2)+'\n').encode(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-new-sources', action='store_true')
    args = parser.parse_args()
    values = build()
    if args.write_new_sources:
        for path, raw in values.items():
            if path.exists():
                if path.read_bytes() != raw:
                    raise ValueError('Existing new target differs; no overwrite: '+str(path))
            else:
                with path.open('xb') as stream:
                    stream.write(raw)
    print(json.dumps(dict(source_only=True, outputs={str(p): hashlib.sha256(r).hexdigest()
        for p, r in values.items()}, live_calls=0)))


if __name__ == '__main__':
    main()
