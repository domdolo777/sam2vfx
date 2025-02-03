import { useEffect, useCallback, useRef, useState } from 'react';
import axios from 'axios';

// Create an axios instance without a hard-coded baseURL so that relative URLs work
const api = axios.create({
  // baseURL is omitted so that the proxy in package.json routes requests properly.
});

const ConcurrentEffectsManager = ({
  objects,
  selectedObjectId,
  effectPreviewMode,
  onEffectUpdate,
  videoId,
  currentFrameIndex,
  effectsMode
}) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const pendingRequestRef = useRef(null);
  const lastAppliedStateRef = useRef(null);
  const parameterUpdateTimeoutRef = useRef(null);
  const initialRenderRef = useRef(true);
  const processingFramesRef = useRef(new Set());
  const originalFramesRef = useRef({});

  // This function updates the effect state by sending a POST request to /apply_effects.
  // The key update is converting the original_frame_hash to a string.
  const handleParameterUpdate = useCallback(async (effectData) => {
    if (isProcessing) return;

    try {
      setIsProcessing(true);

      // Cancel any pending request if it exists.
      if (pendingRequestRef.current?.cancel) {
        pendingRequestRef.current.cancel();
      }
      if (parameterUpdateTimeoutRef.current) {
        clearTimeout(parameterUpdateTimeoutRef.current);
      }

      const CancelToken = axios.CancelToken;
      const source = CancelToken.source();
      pendingRequestRef.current = source;

      // Create a unique key for the current frame.
      const frameKey = `${currentFrameIndex}`;
      if (processingFramesRef.current.has(frameKey)) return;
      processingFramesRef.current.add(frameKey);

      // Store a reference value for the frame if it doesn’t already exist.
      // (We use Date.now() as a simple hash; since Date.now() returns a number,
      // we will later convert it to a string.)
      if (!originalFramesRef.current[frameKey]) {
        originalFramesRef.current[frameKey] = Date.now();
      }

      // IMPORTANT: Convert the original frame hash to a string.
      const originalFrameHashStr = originalFramesRef.current[frameKey].toString();

      const updateData = {
        video_id: videoId,
        frame_idx: currentFrameIndex,
        obj_id: selectedObjectId,
        effects: effectData.effects || [],
        feather_params: effectData.feather_params || {},
        preview_mode: effectPreviewMode,
        reset_frame: true,
        // Send the hash as a string to satisfy the backend schema
        original_frame_hash: originalFrameHashStr
      };

      // Create a hash of the updateData (excluding the original_frame_hash conversion issue)
      const stateHash = JSON.stringify({
        effects: updateData.effects,
        feather_params: updateData.feather_params,
        frame_idx: updateData.frame_idx,
        preview_mode: updateData.preview_mode
      });

      // If this state is the same as the last applied one, skip the update.
      if (lastAppliedStateRef.current === stateHash) {
        processingFramesRef.current.delete(frameKey);
        return;
      }

      // Send the request using a relative URL (which will be forwarded via your proxy).
      const response = await api.post('/apply_effects', updateData, {
        cancelToken: source.token
      });

      if (response.status === 200) {
        lastAppliedStateRef.current = stateHash;
        if (onEffectUpdate) {
          onEffectUpdate();
        }
      }
    } catch (error) {
      if (!axios.isCancel(error)) {
        console.error('Error applying effects:', error?.response?.data || error);
      }
    } finally {
      setIsProcessing(false);
      const frameKey = `${currentFrameIndex}`;
      processingFramesRef.current.delete(frameKey);
      pendingRequestRef.current = null;
    }
  }, [videoId, currentFrameIndex, selectedObjectId, effectPreviewMode, onEffectUpdate, isProcessing]);

  // Trigger a parameter update when frame index or effect mode changes.
  useEffect(() => {
    lastAppliedStateRef.current = null;
    if (effectsMode) {
      const selectedObject = objects.find(obj => obj.id === selectedObjectId);
      if (selectedObject) {
        const effectData = {
          effects: selectedObject.effects,
          feather_params: selectedObject.featherParams
        };
        handleParameterUpdate(effectData);
      }
    }
  }, [currentFrameIndex, effectsMode, objects, selectedObjectId, handleParameterUpdate]);

  // Handle the initial frame update.
  useEffect(() => {
    if (!selectedObjectId) return;
    const selectedObject = objects.find(obj => obj.id === selectedObjectId);
    if (!selectedObject) return;
    if (currentFrameIndex === 0 && initialRenderRef.current) {
      initialRenderRef.current = false;
      const initTimeout = setTimeout(() => {
        handleParameterUpdate({
          effects: selectedObject.effects,
          feather_params: selectedObject.featherParams
        });
      }, 100);
      return () => clearTimeout(initTimeout);
    }
  }, [objects, selectedObjectId, currentFrameIndex, handleParameterUpdate]);

  // Cleanup on unmount.
  useEffect(() => {
    let processingFramesCurrent = processingFramesRef.current;
    let parameterTimeoutCurrent = parameterUpdateTimeoutRef.current;
    let pendingRequestCurrent = pendingRequestRef.current;
    return () => {
      if (processingFramesCurrent) {
        [...processingFramesCurrent].forEach(key => {
          processingFramesCurrent.delete(key);
        });
      }
      if (parameterTimeoutCurrent) {
        clearTimeout(parameterTimeoutCurrent);
      }
      if (pendingRequestCurrent?.cancel) {
        pendingRequestCurrent.cancel();
      }
    };
  }, []);

  return null;
};

export default ConcurrentEffectsManager;
