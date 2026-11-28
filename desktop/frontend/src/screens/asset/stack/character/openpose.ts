// How a skeleton frame is drawn for a video model (pipeline#14): OpenPose-18, as runners/skeleton_video.py draws it,
// with the Vision joint for each index, its limbs and their colours. Shared by scripts/skeleton-cycle.ts (the
// skeleton video LTX-2.3 takes) and the Stack stage's side-by-side compare (pipeline#8), so both draw the same thing.
import type { V3 } from "./rig";
import type { JointName } from "./skeleton";

export const OPENPOSE: JointName[] = ["head_joint", "neck_1_joint", "right_shoulder_1_joint", "right_forearm_joint", "right_hand_joint",
  "left_shoulder_1_joint", "left_forearm_joint", "left_hand_joint", "right_upLeg_joint", "right_leg_joint",
  "right_foot_joint", "left_upLeg_joint", "left_leg_joint", "left_foot_joint", "right_eye_joint",
  "left_eye_joint", "right_ear_joint", "left_ear_joint"];
export const OPENPOSE_LIMBS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
  [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]];
export const OPENPOSE_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
  [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
  [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]];

/** Joints the camera sees (it looks along +y): an eye or ear on the far side of the head is left out, as a pose
 * detector would. */
export function seenJoints(j: Record<JointName, V3>): Partial<Record<JointName, V3>> {
  const s: Partial<Record<JointName, V3>> = { ...j };
  const mid = (j.left_ear_joint[1] + j.right_ear_joint[1]) / 2;
  const reach = Math.abs(j.left_ear_joint[1] - j.right_ear_joint[1]) / 2;
  for (const n of ["left_eye_joint", "right_eye_joint", "left_ear_joint", "right_ear_joint"] as JointName[])
    if (j[n][1] - mid > 0.35 * reach) delete s[n];
  return s;
}
