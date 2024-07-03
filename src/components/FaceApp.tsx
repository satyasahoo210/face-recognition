import "@tensorflow/tfjs-backend-webgl";

import _ from "lodash";
import { FC, ReactElement, useEffect, useRef, useState } from "react";
import { CircularProgress } from "@mui/material";
import { Box } from "@mui/system";
import {
  Face,
  MediaPipeFaceMeshMediaPipeEstimationConfig,
  createDetector,
} from "@tensorflow-models/face-landmarks-detection";
import React from "react";
import { getKeypointsByContour } from "../utils/keypoint-utils";
import { BoundingBox } from "@tensorflow-models/face-landmarks-detection/dist/shared/calculators/interfaces/shape_interfaces";
import { detectorConfig, model } from "../utils/constants";
import { useModels } from "../utils/ModelContext";

type EyeState = "closed" | "open";

export type Data = {
  eyeState: EyeState;
  bbox: BoundingBox;
};

interface Config {
  modelConfig: MediaPipeFaceMeshMediaPipeEstimationConfig;
  camera?: {
    constrains?: MediaStreamConstraints;
  };
  renderMesh?: boolean;
}

interface Options {
  config?: Config;
  onBlink?: (data: Data) => void;
  onNoFace?: () => void;
  onMultipleFaces?: () => void;
  className?: string;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  videoRef: React.RefObject<HTMLVideoElement>;
  stopAnimation?: boolean;
  start?: boolean;
  boundingBox?: boolean;
  faceOval?: boolean;
}

const defaultOptions: Partial<Options> = {
  config: {
    modelConfig: {
      flipHorizontal: false,
      staticImageMode: false,
    },
    camera: {
      constrains: {
        audio: false,
        video: {
          facingMode: "user",
        },
      },
    },
  },
  className: "face-app",
};
const RED = "#FF2C35";
const GREEN = "#32EEDB";

const distance = (a: number[], b: number[]) =>
  Math.sqrt(Math.pow(a[0] - b[0], 2) + Math.pow(a[1] - b[1], 2));
const drawPath = (
  ctx: CanvasRenderingContext2D,
  points: number[][],
  closePath: boolean
) => {
  const region = new Path2D();
  region.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    const point = points[i];
    region.lineTo(point[0], point[1]);
  }

  if (closePath) {
    region.closePath();
  }
  ctx.stroke(region);
};

const FaceApp: FC<Options> = (props): ReactElement => {
  const options: Options = _.merge(defaultOptions, props);

  const [stream, setStream] = useState<MediaStream>();
  const [videoWidth, setVideoWidth] = useState(0);
  const [videoHeight, setVideoHeight] = useState(0);
  const [ctx, setCtx] = useState<CanvasRenderingContext2D>();
  const [ctxMask, setCtxMask] = useState<CanvasRenderingContext2D>();
  const [loading, setLoading] = useState<string>("");
  const {detector, setDetector} = useModels()

  const canvasMask = useRef<HTMLCanvasElement>(null);
  const canvas = props.canvasRef;
  const video = props.videoRef;

  let rafID = -1;

  let eyeGap = {
    left: [0],
    right: [0],
  };
  let eyeState: EyeState = "open";

  const processPrediction = (faces: Face[]) => {
    const isClosed = (...diff: number[]) => diff.every((e) => e > 5);

    if (faces.length > 1) {
      options.onMultipleFaces && options.onMultipleFaces();
      return;
    }

    if (!faces.length) {
      options.onNoFace && options.onNoFace();
      return;
    }

    let face: Face = faces[0];
    const leftEye = getKeypointsByContour("leftEye", face.keypoints).map(
      (pt) => [pt.x, pt.y]
    );
    let leftEyeLower: number[][] = leftEye.slice(
        0,
        Math.floor(leftEye.length / 2)
      ),
      leftEyeUpper: number[][] = leftEye.slice(Math.ceil(leftEye.length / 2));

    const rightEye = getKeypointsByContour("rightEye", face.keypoints).map(
      (pt) => [pt.x, pt.y]
    );
    let rightEyeLower: number[][] = rightEye.slice(
        0,
        Math.floor(rightEye.length / 2)
      ),
      rightEyeUpper: number[][] = rightEye.slice(
        Math.ceil(rightEye.length / 2)
      );

    let leftEyeLowerMiddle = leftEyeLower[Math.floor(leftEyeLower.length / 2)];
    let leftEyeUpperMiddle = leftEyeUpper[Math.floor(leftEyeUpper.length / 2)];

    let rightEyeLowerMiddle =
      rightEyeLower[Math.floor(rightEyeLower.length / 2)];
    let rightEyeUpperMiddle =
      rightEyeUpper[Math.floor(rightEyeUpper.length / 2)];

    let leftGap = distance(leftEyeLowerMiddle, leftEyeUpperMiddle);
    let rightGap = distance(rightEyeLowerMiddle, rightEyeUpperMiddle);

    let leftAvgGap = eyeGap.left.reduce((a, c) => a + c) / eyeGap.left.length;
    let rightAvgGap =
      eyeGap.right.reduce((a, c) => a + c) / eyeGap.right.length;

    let leftDiff = leftAvgGap - leftGap;
    let rightDiff = rightAvgGap - rightGap;

    let currentEyeState: EyeState = isClosed(leftDiff, rightDiff)
      ? "closed"
      : "open";

    if (currentEyeState === "closed" && currentEyeState !== eyeState) {
      options.onBlink &&
        options.onBlink({ eyeState: currentEyeState, bbox: face.box });
    }

    eyeGap.left = [leftGap, ...eyeGap.left.slice(0, 9)];
    eyeGap.right = [rightGap, ...eyeGap.right.slice(0, 9)];
    eyeState = currentEyeState;
  };

  async function setUp() {
    setLoading("Configuring Camera");
    await setupCamera();
    (video.current as HTMLVideoElement).play();
    setVideoWidth((video.current as HTMLVideoElement)?.videoWidth || 0);
    setVideoHeight((video.current as HTMLVideoElement)?.videoHeight || 0);
  }

  async function setupCamera() {
    try {
      navigator.mediaDevices
        .getUserMedia(options.config?.camera?.constrains)
        .then((_stream) => {
          setStream(_stream);
          (video.current as HTMLVideoElement).srcObject = _stream;
        });

      return new Promise((resolve) => {
        (video.current as HTMLVideoElement).onloadedmetadata = () => {
          resolve(video.current);
        };
      });
    } catch (error) {
      console.error("Error accessing the camera:", error);
      alert(
        "Camera access is required to use this application. Please allow camera access."
      );
      throw error;
    }
  }

  async function renderPrediction() {
    if (!!detector && video.current instanceof HTMLVideoElement) {
      try {
        const faces: Face[] = await detector.estimateFaces(
          video.current,
          options.config?.modelConfig
        );
        (ctx as CanvasRenderingContext2D).drawImage(
          video.current,
          0,
          0,
          videoWidth,
          videoHeight,
          0,
          0,
          videoWidth,
          videoHeight
        );

        processPrediction(faces);

        if (!!ctxMask) {
          ctxMask.clearRect(0, 0, videoWidth, videoHeight);
          faces.forEach((face: Face) => {
            if (props.boundingBox) {
              ctxMask.strokeStyle = RED;
              ctxMask.lineWidth = 1;

              const box = face.box;
              drawPath(
                ctxMask,
                [
                  [box.xMin, box.yMin],
                  [box.xMax, box.yMin],
                  [box.xMax, box.yMax],
                  [box.xMin, box.yMax],
                ],
                true
              );
            }

            if (props.faceOval) {
              const silhouette: number[][] = getKeypointsByContour(
                "faceOval",
                face.keypoints
              ).map((pt) => [pt.x, pt.y]);
              ctxMask.strokeStyle = GREEN;
              ctxMask.fillStyle = GREEN;

              for (let i = 0; i < silhouette.length; i++) {
                const x = silhouette[i][0];
                const y = silhouette[i][1];

                ctxMask.beginPath();
                ctxMask.arc(x, y, 1 /* radius */, 0, 2 * Math.PI);
                ctxMask.fill();
              }

              drawPath(ctxMask, silhouette, true);
            }
          });
        }
      } catch {}
    }
    if (!props.stopAnimation) {
      rafID = requestAnimationFrame(renderPrediction);
    } else {
      cancelAnimationFrame(rafID);
    }
  }

  useEffect(() => {
    (async function () {
      if (props.start && video.current && video.current.readyState < 2) {
        createDetector(model, detectorConfig).then((_detector) => {
          setDetector?.(_detector);
        });
        await setUp();
        setLoading("");
      }
    })();

    return () => {
      if (!!stream) {
        stream.getTracks().forEach((track) => {
          track.stop();
        });
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.start]);

  useEffect(() => {
    if (canvas.current instanceof HTMLCanvasElement) {
      canvas.current.width = videoWidth;
      canvas.current.height = videoHeight;

      let ctx = canvas.current.getContext("2d") as CanvasRenderingContext2D;
      /* block For mirror effect */
      ctx.translate(videoWidth, 0);
      ctx.scale(-1, 1);
      /* endblock */

      setCtx(ctx);
    }
  }, [canvas, videoHeight, videoWidth]);

  useEffect(() => {
    if (canvasMask.current instanceof HTMLCanvasElement) {
      canvasMask.current.width = videoWidth;
      canvasMask.current.height = videoHeight;

      let ctx = canvasMask.current.getContext("2d") as CanvasRenderingContext2D;
      /* block For mirror effect */
      ctx.translate(videoWidth, 0);
      ctx.scale(-1, 1);
      /* endblock */

      setCtxMask(ctx);
    }
  }, [canvasMask, videoHeight, videoWidth]);

  useEffect(() => {
    !loading && renderPrediction();

    return () => cancelAnimationFrame(rafID);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loading]);

  return (
    <div className={options.className}>
      <Box
        sx={{
          position: "relative",
          backgroundColor: "#000000",
          borderRadius: 2,
          height: videoHeight || 480,
          width: videoWidth || 1000,
        }}
      >
        {!!loading && (
          <Box
            sx={{
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              gap: 3,
              color: "#ffffff",
              backgroundColor: "rgb(0 0 0 / 70%)",
              position: "absolute",
              top: 0,
              bottom: 0,
              right: 0,
              left: 0,
              zIndex: 10,
            }}
          >
            <CircularProgress size={20} />
            {loading}
          </Box>
        )}
        <canvas
          ref={canvas}
          height={videoHeight}
          width={videoWidth}
          style={{ borderRadius: 2 }}
        />
        <canvas
          ref={canvasMask}
          height={videoHeight}
          width={videoWidth}
          style={{ position: "absolute", left: 0 }}
        />
        <video
          ref={video}
          height={videoHeight}
          width={videoWidth}
          playsInline
          style={{ display: "none" }}
        />
      </Box>
    </div>
  );
};

export default FaceApp;
