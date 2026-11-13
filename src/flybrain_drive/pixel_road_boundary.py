"""Fixed-mount, body-frame road-edge candidate detector for archived AC images."""
import numpy as np

from .flyvis_camera import VIEW_ORDER


class PixelRoadBoundary:
    def __init__(self, mount):
        self.mount = mount
        self.lateral = np.linspace(-14., 14., 281)
        self.near = np.arange(0., 14., 2.)
        self.far = np.array([10., 15., 20.])
        self.near_lut = self._sampling(mount, self.near)
        self.far_lut = self._sampling(mount, self.far)

    def _sampling(self, mount, stations, pitch=0., roll=0.):
        x, y = np.meshgrid(stations, self.lateral, indexing="ij")
        points = np.stack((x, y, -np.tan(pitch)*x-np.tan(roll)*y), axis=-1)
        lookups = {}
        for name in VIEW_ORDER:
            camera = mount["views"][name]
            delta = points - camera["origin_body_m"]
            depth = delta @ camera["forward_body"]
            safe = np.where(depth > 0, depth, 1.)
            u = .5 + .5*camera["projection_x"]*(delta @ camera["right_body"])/safe
            v = .5 - .5*camera["projection_y"]*(delta @ camera["up_body"])/safe
            valid = (depth > 0) & (u >= .02) & (u <= .98) & (v >= .02) & (v <= .98)
            lookups[name] = u, v, valid
        return lookups

    def _classify(self, images, lookups, reject_disagreement=False):
        votes = np.zeros_like(lookups[VIEW_ORDER[0]][2], dtype=int)
        seen = np.zeros_like(votes)
        for name, (u, v, valid) in lookups.items():
            height, width = images[name].shape[:2]
            b, g, r = images[name][np.rint(v[valid]*(height-1)).astype(int),
                                   np.rint(u[valid]*(width-1)).astype(int)].astype(float).T
            votes[valid] += (g-r > 6) & (g-b > 12) & (g > 40)
            seen[valid] += 1
        visible = seen > 0
        if reject_disagreement:
            visible &= (votes == 0) | (votes == seen)
        return (votes > seen/2) & visible, visible

    def _sections(self, stations, grass, visible, near):
        sections = []
        center = 0.
        for x, green, supported in zip(stations, grass, visible, strict=True):
            start = int(np.argmin(abs(self.lateral-center)))
            if near:
                def is_edge(q, side):
                    offsets = np.arange(4)*side
                    inside, outside = q-offsets-side, q+offsets
                    return (green[q] and supported[inside].all() and supported[outside].all()
                            and green[outside].sum() >= 3 and not green[inside].any())
            else:
                kernel = np.ones(5)
                vegetation = np.convolve(green,kernel,"same") >= 4
                pavement = np.convolve(supported & ~green,kernel,"same") >= 4

                def is_edge(q, side):
                    return vegetation[q] and pavement[q-side*np.arange(3,8)].sum() >= 3

            edges = []
            for side, scan in ((-1, range(start-1, 5, -1)), (1, range(start+1, len(self.lateral)-6))):
                edge = None
                if near or pavement[start]:
                    gap = 0
                    for q in scan:
                        gap = 0 if supported[q] else gap+1
                        if not near and gap > 2:
                            break
                        if not gap and is_edge(q, side):
                            edge = float(self.lateral[q])
                            break
                edges.append(edge)
            right, left = edges
            valid = right is not None and left is not None and 3. < left-right < 20.
            sections.append(dict(forward_x_m=float(x), left_y_m=left, right_y_m=right, valid=valid))
            if valid:
                center = .5*(left+right)
        return sections

    def detect(self, images, *, tilt=None, reject_disagreement=False):
        if set(images) != set(VIEW_ORDER):
            raise ValueError("Complete four-view image bundle required")
        near_lut, far_lut = self.near_lut, self.far_lut
        if tilt is not None:
            pitch, roll = tilt
            if not np.isfinite([pitch,roll]).all() or max(abs(pitch),abs(roll)) >= np.pi/4:
                raise ValueError("Finite body pitch and roll in radians within 45 degrees required")
            near_lut = self._sampling(self.mount,self.near,pitch,roll)
            far_lut = self._sampling(self.mount,self.far,pitch,roll)
        near = self._sections(self.near, *self._classify(images,near_lut,reject_disagreement), near=True)
        far = self._sections(self.far, *self._classify(images,far_lut,reject_disagreement), near=False)
        return near[2:] + far[1:]
