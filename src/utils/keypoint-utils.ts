import { Keypoint, util } from "@tensorflow-models/face-landmarks-detection";
import { model } from "./constants";

type ContourNames = "lips" | "leftEye" | "leftEyebrow" | "leftIris" | "rightEye" | "rightEyebrow" | "rightIris" | "faceOval"

const INDICES = util.getKeypointIndexByContour(model)

export const getKeypointsByContour = (contourName: ContourNames, keypoints: Keypoint[]) => {
    return INDICES[contourName].map(idx => keypoints[idx])
}
