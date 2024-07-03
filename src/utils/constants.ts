import { MediaPipeFaceMeshMediaPipeModelConfig } from "@tensorflow-models/face-landmarks-detection/dist/mediapipe/types";
import { SupportedModels } from "@tensorflow-models/face-landmarks-detection/dist/types";

export const SIMILARITY_THRESHOLD = 0.7;

export const model = SupportedModels.MediaPipeFaceMesh

export const detectorConfig: MediaPipeFaceMeshMediaPipeModelConfig = {
  runtime: 'mediapipe', // or 'tfjs'
  refineLandmarks: true,
  maxFaces: 1,
  solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh',
}

export const INCEPTION_MODEL_URL = "https://kaggle.com/models/google/inception-v1/TfJs/classification/1"