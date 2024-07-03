import * as tf from "@tensorflow/tfjs";
import { FaceLandmarksDetector } from '@tensorflow-models/face-landmarks-detection';
import React, { createContext, useContext, useState, ReactNode } from 'react';

type ModelContextType = {
    detector?: FaceLandmarksDetector,
    setDetector?: React.Dispatch<React.SetStateAction<FaceLandmarksDetector | undefined>>
    encoder?: tf.GraphModel
    setEncoder?: React.Dispatch<React.SetStateAction<tf.GraphModel | undefined>>
    disposeAll?: () => void
    disposeDetector?: () => void
    disposeEncoder?: () => void
};

const ModelContext = createContext<ModelContextType>({});

const ModelProvider = ({ children }: {children?: ReactNode}) => {

    const [detector, setDetector] = useState<FaceLandmarksDetector>()
    const [encoder, setEncoder] = useState<tf.GraphModel>();

    const disposeDetector = () => {
        if (detector != null) {
            detector.dispose();
        }
    }

    const disposeEncoder = () => {
        if (encoder != null) {
            encoder.dispose();
        }
    }

    const disposeAll = () => {
        disposeEncoder();
        disposeDetector();
    }

    return (
        <ModelContext.Provider value={{detector, encoder, setDetector, setEncoder, disposeEncoder, disposeDetector, disposeAll}}>
            {children}
        </ModelContext.Provider>
    )
}



const useModels = () => {
    return useContext(ModelContext)
}

export { ModelProvider, useModels}