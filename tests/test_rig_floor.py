"""rig_pred.floor_pass on the sheep (out/mlg-sheep, when it is there): a lying clip rests its trunk on the floor, not
on its folded hooves, and Eating lowers the head with the neck while the trunk stands (Stephen: "I was thinking it
would just move the head down to eat")."""
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

    def test_eating_keeps_the_trunk_standing_and_lowers_the_head(self) -> None:
        # before: the horse's 63 degree neck-base turn folded the sheep's chest down onto its front legs (the belly
        # 0.07 of the height under Idle's); now the neck base keeps half its turn (NECK_BASE_KEEP), the trunk behind it
        # none
        idle = self.belly("Idle")
        self.assertGreater(self.belly("Eating").min(), idle.min() - 0.04)
        c = self.by["Eating"]
        f = len(c["times"]) // 2
        for n in ("spine_1", "spine_2", "spine_3"):
            self.assertLess(np.degrees(R.from_quat(c["rot"][self.node[n]][f]).magnitude()), 1.0, n)
        reach = self.report["Eating"]["head_reach"]
        self.assertEqual(reach["pitch_max_deg"], 0.0)
        self.assertGreater(reach["bend_max_deg"], 0.0)
        head = np.isin(np.array(self.fk.joints)[self.J[np.arange(len(self.J)), self.W.argmax(1)]],
                       [self.node[n] for n in self.node if n.startswith("head") or n in ("spine_5", "spine_6")])
        self.assertLess(self.pose("Eating", f)[head, 1].min(), self.pose("Idle", 0)[head, 1].min() - 0.1 * self.H)
        self.assertGreater(self.pose("Eating", f)[head, 1].min(), self.floor - 0.005 * self.H)


if __name__ == "__main__":
    unittest.main()
