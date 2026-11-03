"""Read-only, synchronized actor/observer replay; no video codec dependency."""

import base64
import html
import json
import struct
import zlib

import numpy as np

from .env import BODY_CHANNELS, BODY_SCALE
from .evidence import write_json
from .graph import sha256


def png_data(image):
    """Encode lossless RGB pixels using the standard PNG container."""
    pixels = np.asarray(image, dtype=np.uint8)
    height, width, channels = pixels.shape
    if channels != 3:
        raise ValueError("Replay requires RGB pixels")

    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    rows = b"".join(b"\x00" + row.tobytes() for row in pixels)
    encoded = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


class PolicyReplay:
    def __init__(self, env, directory, title, every=4):
        if every < 1:
            raise ValueError("record_every must be positive")
        self.env, self.directory, self.title, self.every = env, directory, title, every
        directory.mkdir()
        self.renderer = env.mj.Renderer(env.model, 128, 128)
        self.camera = env.mj.MjvCamera()
        self.camera.type = env.mj.mjtCamera.mjCAMERA_FREE
        self.camera.distance = 8.0
        self.camera.azimuth = 135.0
        self.camera.elevation = -25.0
        self.frames, self.samples = [], []

    def capture(self, tick, observation, info, action=None, final=False):
        if tick % self.every and not final:
            return None
        env = self.env
        self.camera.lookat[:] = env.data.qpos[:3] + np.array([0, 0, 0.4])
        self.renderer.update_scene(env.data, camera=self.camera)
        env.add_render_markers(self.renderer.scene)
        observer = self.renderer.render().copy()
        physical = observation["body"] * BODY_SCALE
        self.frames.append(
            {
                "tick": tick,
                "time": info["time"],
                "reason": info["reason"],
                "action": action.tolist() if action is not None else None,
                "camera_age": observation["camera_age"].tolist(),
                "camera_valid": observation["camera_valid"].tolist(),
                "speed_encoder": float(physical[24]),
                "gyro_yaw": float(physical[2]),
                "sensor_valid": observation["body_valid"].tolist(),
                "lateral_error": info["truth"]["lateral_error"],
                "progress": info["truth"]["progress_3d"],
                "left": png_data(observation["images"][0]),
                "right": png_data(observation["images"][1]),
                "observer": png_data(observer),
                "issued_control": None,
            }
        )
        self.samples.append(
            {
                **{key: value.copy() for key, value in observation.items()},
                "observer": observer,
                "camera_source": env.camera_source_images.copy(),
                "time": info["time"],
                "tick": tick,
                "qpos": env.data.qpos.copy(),
                "qvel": env.data.qvel.copy(),
                "action": action.copy() if action is not None else np.zeros(
                    env.action_space.shape, np.float32
                ),
                "action_valid": action is not None,
                "applied_control": np.zeros_like(env.data.ctrl),
            }
        )
        if "motion" in observation:
            strength = np.clip(observation["motion"][..., 2:].sum(-1), 0, 1)
            heatmap = np.repeat(np.rint(strength * 255)[..., None], 3, axis=-1).astype(np.uint8)
            self.frames[-1].update(
                source_left=png_data(env.camera_source_images[0]),
                source_right=png_data(env.camera_source_images[1]),
                motion_left=png_data(heatmap[0]), motion_right=png_data(heatmap[1]),
            )
        return len(self.frames) - 1

    def after_step(self, index, info):
        if index is not None:
            self.frames[index]["issued_control"] = info["control"]
            self.frames[index]["transition_end_time"] = info["time"]
            self.frames[index]["transition_reason"] = info["reason"]
            self.samples[index]["applied_control"] = self.env.data.ctrl.copy()

    def finish(self):
        raw_path = self.directory / "samples.npz"
        np.savez_compressed(
            raw_path,
            **{key: np.stack([row[key] for row in self.samples]) for key in self.samples[0]},
            ray_directions_camera_local=self.env.sampler.rays,
            ray_valid=self.env.sampler.valid,
        )
        payload = json.dumps(self.frames, ensure_ascii=False, allow_nan=False)
        payload = payload.replace("<", "\\u003c")
        page = _PAGE.replace("__TITLE__", html.escape(self.title)).replace("__FRAMES__", payload)
        page_path = self.directory / "replay.html"
        page_path.write_text(page, encoding="utf-8")
        metadata = {
            "frames": len(self.frames),
            "record_every_control_ticks": self.every,
            "recording_period_seconds": self.every * self.env.dt,
            "observer_resolution": [128, 128],
            "actor_resolution": [self.env.image_size, self.env.image_size],
            "camera_source_resolution": [self.env.sampler.source_size] * 2,
            "visual_schema": self.env.visual_schema,
            "actuation_mode": self.env.control.actuation_mode,
            "time_contract": "images/body/action at t; issued control drives transition t to t+dt",
            "terminal_frame": "action_valid=false; no subsequent control applied",
            "body_channels": BODY_CHANNELS,
            "body_scale": BODY_SCALE.tolist(),
            "observer_contract": "diagnostic ground-truth view only; never a policy input",
            "files_sha256": {p.name: sha256(p) for p in (raw_path, page_path)},
        }
        write_json(self.directory / "recording.json", metadata)
        return metadata

    def close(self):
        self.renderer.close()


_PAGE = r'''<!doctype html>
<html lang="ko"><meta charset="utf-8"><title>__TITLE__</title>
<style>
body{background:#111923;color:#e6edf5;font:16px system-ui;margin:32px auto;max-width:1100px}
h1{font-size:24px}small{color:#aabbd0}button,select{font:inherit;padding:6px;background:#24344a;color:inherit}
.views{display:flex;gap:16px;flex-wrap:wrap}.views figure{margin:12px 0}
.views img{width:256px;height:256px}
.eyes{image-rendering:pixelated}figcaption{margin:6px 0}input{width:100%}
pre{white-space:pre-wrap;background:#1b2839;padding:16px;font:15px ui-monospace,monospace}
canvas{width:100%;height:150px;background:#1b2839}p{line-height:1.5}
</style><h1>__TITLE__</h1>
<p>정책이 받은 좌우 원본 입력과 외부 진단 화면. 외부 화면은 정책에 제공하지 않습니다.<br>
<small>합성 노면·보정 전 차량. 입력은 t 시점,
명령은 t→t+dt 구간입니다. 카메라·몸 감각은 기존 정책 입력 그대로입니다.</small></p>
<button id="play">재생</button> <select id="speed"><option value="0.5">0.5배</option>
<option value="1" selected>1배</option><option value="2">2배</option></select>
<input id="seek" type="range" min="0" value="0" step="1">
<div class="views"><figure><figcaption>좌측 정책 RGB 입력</figcaption>
<img id="left" class="eyes"></figure>
<figure><figcaption>우측 정책 RGB 입력</figcaption><img id="right" class="eyes"></figure>
<figure><figcaption>외부 관찰 · 128×128</figcaption><img id="observer"></figure></div>
<pre id="metrics"></pre><p>주행 중 발행한 후륜 토크 명령(N·m): 청록=좌, 주황=우</p>
<canvas id="plot" width="1000" height="150"></canvas>
<details id="extra"><summary>렌더링 원본과 운동 응답 보기</summary><div class="views">
<figure><figcaption>좌측 렌더링 원본</figcaption><img id="source_left"></figure>
<figure><figcaption>우측 렌더링 원본</figcaption><img id="source_right"></figure>
<figure><figcaption>좌측 운동 응답 크기</figcaption><img id="motion_left" class="eyes"></figure>
<figure><figcaption>우측 운동 응답 크기</figcaption><img id="motion_right" class="eyes"></figure>
</div></details>
<script>
const frames=__FRAMES__;
const seek=document.getElementById('seek'),play=document.getElementById('play');
seek.max=frames.length-1;let timer=null,playing=false;
function draw(){
 const i=Number(seek.value),f=frames[i],c=f.issued_control;
 for(const name of ['left','right','observer'])document.getElementById(name).src=f[name];
 document.getElementById('extra').hidden=!f.source_left;
 if(f.source_left)for(const name of ['source_left','source_right','motion_left','motion_right'])
  document.getElementById(name).src=f[name];
 const n=x=>Number(x).toFixed(3),a=f.action?f.action.map(n).join(', '):'없음 · 종료 프레임';
 const torques=c?c.rear_torque.map(n).join(', '):'없음';
 const sensed=(x,index)=>f.sensor_valid[index]>0?n(x):'무효/차단';
 const ref=c&&f.sensor_valid[24]>0?n(f.speed_encoder*c.curvature_ref):'관측 불가';
 const labels=frames.find(x=>x.action)?.action.length===3?
  '속도, 곡률, 추가 요모멘트':'속도, 곡률';
 const mz=c&&'residual_yaw_moment_requested' in c?
  `추가 요모멘트 요청=${n(c.residual_yaw_moment_requested)} N·m · `+
  `명령 토크로 계산한 요모멘트=${n(c.yaw_moment_from_commanded_torques)} N·m\n`:'';
 document.getElementById('metrics').textContent=
  `t=${n(f.time-frames[0].time)} s · tick=${f.tick} · ${f.reason}\n`+
  `action [${labels}]=[${a}]\n후륜 토크 명령 [좌, 우]=[${torques}] N·m\n`+mz+
  `정책 encoder speed=${sensed(f.speed_encoder,24)} m/s · `+
  `gyro yaw=${sensed(f.gyro_yaw,2)} rad/s\n`+
  `하위 제어기 yaw reference=v_measured×κ_ref=${ref} rad/s\n`+
  `camera age [좌, 우]=[${f.camera_age.map(n).join(', ')}] s · valid=[${f.camera_valid}]\n`+
  `진단 정답: 횡오차=${n(f.lateral_error)} m · 진행=${n(f.progress)} m\n`+
  `구간 종료=${f.transition_reason||f.reason}`;
 const ctx=document.getElementById('plot').getContext('2d'),w=1000,h=150;
 ctx.clearRect(0,0,w,h);
 const max=Math.max(1,...frames.flatMap(x=>x.issued_control?x.issued_control.rear_torque:[]));
 const end=frames.at(-1).time-frames[0].time||1;
 for(let side=0;side<2;side++){
  ctx.strokeStyle=['#60dfce','#ffb56b'][side];ctx.beginPath();let started=false;
  for(const r of frames){if(!r.issued_control)continue;
   const x=(r.time-frames[0].time)/end*w,y=h-12-r.issued_control.rear_torque[side]/max*(h-24);
   if(started)ctx.lineTo(x,y);else{ctx.moveTo(x,y);started=true}}
  ctx.stroke();
 }
 ctx.strokeStyle='#e6edf5';ctx.beginPath();const x=(f.time-frames[0].time)/end*w;
 ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();
}
function advance(){
 if(!playing)return;const i=Number(seek.value);
 if(i>=frames.length-1){playing=false;play.textContent='재생';return}
 const delay=(frames[i+1].time-frames[i].time)*1000/Number(document.getElementById('speed').value);
 timer=setTimeout(()=>{seek.value=i+1;draw();advance()},Math.max(1,delay));
}
play.onclick=()=>{clearTimeout(timer);playing=!playing;play.textContent=playing?'일시정지':'재생';
 if(playing){if(Number(seek.value)>=frames.length-1)seek.value=0;draw();advance()}};
seek.oninput=()=>{clearTimeout(timer);draw();if(playing)advance()};draw();
</script></html>'''
