"""rig_pred.floor_pass on the sheep (out/mlg-sheep, when it is there): a lying clip rests its trunk on the floor, not
on its folded hooves, Walk's body rises and falls smoothly, and Eating brings the head to the floor with a straight
trunk tipped over spread front legs (Stephen: "I was thinking it would just move the head down to eat")."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import rig  # noqa: E402
import rig_plan  # noqa: E402
import rig_pred  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

SHEEP = next((p for p in (ROOT / "out" / "mlg-sheep", Path.home() / "projects" / "ai-asset-pipeline" / "out" / "mlg-sheep")
              if (p / "rig" / "rigged_nofloor.glb").is_file()), ROOT / "out" / "mlg-sheep")
TRUNK = {"root", "hips", "spine_1", "spine_2", "spine_3", "spine_4", "ribcage_front", "ribcage_back"}


def read_clips(doc: dict, binary: bytes, keep: set[str]) -> list[dict]:
    """The GLB's animations in bake_clips' shape (rot keyed by joint node, the root's translation)."""
    joints = doc["skins"][0]["joints"]
    out = []
    for an in doc.get("animations", []):
        if an["name"] not in keep:
            continue
        rot, root, root_t, times = {}, None, None, None
        for ch in an["channels"]:
            smp = an["samplers"][ch["sampler"]]
            times = rig.accessor(doc, binary, smp["input"])[:, 0] if times is None else times
            val = np.asarray(rig.accessor(doc, binary, smp["output"]), np.float32)
            if ch["target"]["path"] == "rotation":
                rot[ch["target"]["node"]] = val
            else:
                root, root_t = ch["target"]["node"], val
        out.append({"name": an["name"], "times": np.asarray(times, np.float32),
                    "rot": {j: rot[j] for j in joints if j in rot}, "root": root, "root_t": root_t})
    return out


@unittest.skipUnless((SHEEP / "rig" / "skinned_raw.glb").is_file(), "no rigged sheep in out/")
class SheepFloorPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        info = json.loads((SHEEP / "rig" / "rig.json").read_text())
        cls.ours, _ = rig.read_glb(SHEEP / "rig" / "skinned_raw.glb")
        nd, nb = rig.read_glb(SHEEP / "rig" / "rigged_nofloor.glb")
        a = nd["meshes"][0]["primitives"][0]["attributes"]
        cls.P = rig.accessor(nd, nb, a["POSITION"]).astype(float)
        cls.J = rig.accessor(nd, nb, a["JOINTS_0"]).astype(np.int64)
        cls.W = rig.accessor(nd, nb, a["WEIGHTS_0"]).astype(float)
        cls.clips = read_clips(nd, nb, {"Idle", "Walk", "Eating", "Sleep"})
        cls.H, cls.floor = float(np.ptp(cls.P[:, 1])), float(cls.P[:, 1].min())
        cls.fk = rig_pred.FK(cls.ours)
        names = {j: cls.ours["nodes"][j]["name"] for j in cls.fk.joints}
        top = np.array(cls.fk.joints)[cls.J[np.arange(len(cls.J)), cls.W.argmax(1)]]
        cls.trunk = np.isin(top, [j for j, n in names.items() if n in TRUNK])
        cls.node = {n: j for j, n in names.items()}
        cls.report = rig_pred.floor_pass(cls.clips, cls.ours, cls.P, cls.J, cls.W, rig.TEMPLATES["quadruped"],
                                         "quadruped", cls.H, np.array([1.0, 0, 0]),
                                         plan=rig.fixed_plan(rig_plan.load(info.get("plan") or "horse"), "quadruped"),
                                         head_reach=True)
        cls.by = {c["name"]: c for c in cls.clips}

    def pose(self, name: str, f: int) -> np.ndarray:
        c = self.by[name]
        pos, rot = self.fk.world({j: c["rot"][j][f].astype(float) for j in c["rot"]}, {c["root"]: c["root_t"][f].astype(float)})
        return rig_pred._skin(self.P, self.J, self.W, self.fk, pos, rot)

    def belly(self, name: str) -> np.ndarray:
        return np.array([(self.pose(name, f)[self.trunk, 1].min() - self.floor) / self.H
                         for f in range(len(self.by[name]["times"]))])

    def test_sleep_rests_its_trunk_on_the_floor(self) -> None:
        # before: the folded hooves were the lowest points and held the belly 0.05 of the height up
        self.assertLess(self.belly("Sleep").max(), 0.02)
        self.assertEqual(self.report["Sleep"]["lying_frames"], len(self.by["Sleep"]["times"]))
        self.assertGreater(self.report["Sleep"]["low_min_after"], -0.03)  # the folded legs sink a little, as the horse's

    def test_standing_clips_are_not_lying(self) -> None:
        self.assertNotIn("lying_frames", self.report["Idle"])
        self.assertNotIn("lying_frames", self.report["Eating"])

    def test_walk_body_rises_and_falls_smoothly(self) -> None:
        # before: grounded frame by frame from the lowest hoof, the body jumped 0.03 of the height between frames
        y = self.by["Walk"]["root_t"][:, 1] / self.H
        self.assertLess(float(np.abs(np.diff(np.r_[y, y[:1]])).max()), 0.015)
        self.assertGreater(float(np.ptp(y)), 0.01)  # still a walk's rise and fall, not a glide
        self.assertGreater(min(self.belly("Walk")), 0.2)  # no hoof under the floor lifts it oddly: a standing belly

    def test_eating_lowers_the_head_to_the_floor_on_spread_front_legs(self) -> None:
        # before: the horse's 63 degree neck-base turn folded the sheep's chest down onto its front legs; then, held
        # standing, its short neck stopped 0.25 of the height up (Stephen: "it does need to be able to move the head
        # down to the floor"). Now the trunk stays straight (spine_1..3 at rest), tips about the hind legs' roots, and
        # the front legs spread and bend under it while the hind legs stand
        c = self.by["Eating"]
        f = len(c["times"]) // 2
        for n in ("spine_1", "spine_2", "spine_3"):
            self.assertLess(np.degrees(R.from_quat(c["rot"][self.node[n]][f]).magnitude()), 1.0, n)
        reach = self.report["Eating"]["head_reach"]
        self.assertGreater(reach["pitch_max_deg"], 5.0)
        self.assertLess(reach["gap_max_after"], 0.03)
        head = np.isin(np.array(self.fk.joints)[self.J[np.arange(len(self.J)), self.W.argmax(1)]],
                       [self.node[n] for n in self.node if n.startswith("head") or n in ("spine_5", "spine_6")])
        low = np.array([(self.pose("Eating", k)[head, 1].min() - self.floor) / self.H for k in range(0, len(c["times"]), 4)])
        self.assertLess(low.min(), 0.02)  # the mouth at the floor (the horse lifts its head a little as it chews)
        self.assertGreater(low.min(), -0.005)

        def joints(name: str, fr: int) -> dict:
            cc = self.by[name]
            pos, _ = self.fk.world({j: cc["rot"][j][fr].astype(float) for j in cc["rot"]},
                                   {cc["root"]: cc["root_t"][fr].astype(float)})
            return pos

        def knee(p: dict, a: str, b: str, k: str) -> float:
            u, v = p[self.node[a]] - p[self.node[b]], p[self.node[k]] - p[self.node[b]]
            return float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))
        pe, pi = joints("Eating", f), joints("Idle", 0)
        for s_ in "lr":
            hind = ("back_leg_upper_" + s_, "back_leg_lower_" + s_, "back_leg_ankle_" + s_)
            self.assertGreater(knee(pe, *hind), knee(pi, *hind) - 10.0)  # the hind legs stand (they crouched to 141)
            front = ("front_humerus_" + s_, "front_leg_upper_" + s_, "front_leg_lower_" + s_)
            self.assertGreater(knee(pe, *front), 100.0)  # bent, not folded under (86 without the spread)
            # spread: the front foot further out from the middle than standing
            foot = self.node["front_leg_foot_" + s_]
            self.assertGreater(abs(float(pe[foot] @ np.array([1.0, 0, 0]))), abs(float(pi[foot] @ np.array([1.0, 0, 0]))) + 0.01 * self.H)

if __name__ == "__main__":
    unittest.main()
