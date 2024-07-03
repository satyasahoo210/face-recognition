import "./App.css";
import * as tf from "@tensorflow/tfjs";

import { useEffect, useRef } from "react";
import FaceApp from "./components/FaceApp";
import { BoundingBox } from "@tensorflow-models/face-landmarks-detection/dist/shared/calculators/interfaces/shape_interfaces";
import { useModels } from "./utils/ModelContext";
import { Face } from "@tensorflow-models/face-landmarks-detection/dist/types";
import { INCEPTION_MODEL_URL, SIMILARITY_THRESHOLD } from "./utils/constants";

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const faceRef = useRef<HTMLCanvasElement>(null);
  const dpRef = useRef<HTMLImageElement>(null);

  const { encoder, setEncoder, detector, disposeAll } = useModels();

  const captureFace = async (bbox: BoundingBox) => {
    if (!encoder) {
      console.warn("encoder not loaded...");
      return false;
    }
    if (!detector) {
      console.warn("detector not loaded...");
      return false;
    }
    if (!(faceRef.current instanceof HTMLCanvasElement)) {
      return false;
    }
    if (!videoRef.current) {
      return false;
    }

    faceRef.current.width = 224;
    faceRef.current.height = 224;

    let ctx = faceRef.current.getContext("2d") as CanvasRenderingContext2D;
    /* block For mirror effect */
    ctx.translate(bbox.width, 0);
    ctx.scale(-1, 1);
    /* endblock */

    ctx.drawImage(
      videoRef.current,
      bbox.xMin,
      bbox.yMin,
      bbox.width,
      bbox.height,
      0,
      0,
      faceRef.current.width,
      faceRef.current.height
    );

    const liveFaceTensor = tf.browser
      .fromPixels(faceRef.current)
      .reshape([-1, 224, 224, 3])
      .asType("float32");

    const liveFaceFeature = normalize(
      encoder.predict(liveFaceTensor) as tf.Tensor
    );

    /* For DP image */
    // TODO: update image url accordingly
    dpRef.current &&
      (dpRef.current.src = "http://localhost:3000/images/dp.jpg");

    const face: Face = (
      await detector.estimateFaces(dpRef.current as HTMLImageElement)
    )?.[0];
    if (!face) {
      console.warn("no face detected in profile picture...");
      return false;
    }

    ctx.drawImage(
      dpRef.current as HTMLImageElement,
      face.box.xMin,
      face.box.yMin,
      face.box.width,
      face.box.height,
      0,
      0,
      faceRef.current.width,
      faceRef.current.height
    );

    const dpFaceTensor = tf.browser
      .fromPixels(faceRef.current)
      .reshape([-1, 224, 224, 3])
      .asType("float32");

    const dpFaceFeature = normalize(encoder.predict(dpFaceTensor) as tf.Tensor);

    const result = tf
      .sqrt(tf.sum(tf.pow(tf.sub(liveFaceFeature, dpFaceFeature), 2)))
      .greaterEqual(SIMILARITY_THRESHOLD)
      .arraySync();
    console.log("similarity match", result, result === 1);
    return result === 1;
  };

  const normalize = (tensor: tf.Tensor) => {
    return tf.div(tensor, tf.sqrt(tf.sum(tf.pow(tensor, 2))));
  };

  useEffect(() => {
    (async function () {
      console.info("Loading encoder ...");
      const _model = await tf.loadGraphModel(
        INCEPTION_MODEL_URL,
        { fromTFHub: true }
      );
      await _model.load();
      setEncoder?.(_model);
      console.info("Encoder loaded ...");
    })();

    return () => {
      disposeAll?.();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="App">
      <FaceApp
        canvasRef={canvasRef}
        videoRef={videoRef}
        start={true}
        boundingBox={true}
        onBlink={({ bbox }) => {
          const timeout = 300; // average time for a human to open eyes after blink
          console.info(`waiting for ${timeout} milliseconds`);

          setTimeout(async () => {
            const result = await captureFace(bbox);
            console.log('Face matched ?? -> ', result);
          }, timeout);
        }}
      />
      <canvas ref={faceRef}></canvas>
      <img src="" alt="" ref={dpRef} />
    </div>
  );
}

export default App;
