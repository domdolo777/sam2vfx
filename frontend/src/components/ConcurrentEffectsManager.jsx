import { useEffect, useCallback, useRef, useState } from 'react';
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000'
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

  const handleParameterUpdate = useCallback(async (effectData) => {
    if (isProcessing) return;
    
    try {
      setIsProcessing(true);

      if (pendingRequestRef.current?.cancel) {
        pendingRequestRef.current.cancel();
      }

      if (parameterUpdateTimeoutRef.current) {
        clearTimeout(parameterUpdateTimeoutRef.current);
      }

      const CancelToken = axios.CancelToken;
      const source = CancelToken.source();
      pendingRequestRef.current = source;

      // Create frame key for tracking
      const frameKey = `${currentFrameIndex}`;
      if (processingFramesRef.current.has(frameKey)) return;
      processingFramesRef.current.add(frameKey);

      // Store original frame reference if not exists
      if (!originalFramesRef.current[frameKey]) {
        originalFramesRef.current[frameKey] = Date.now();
      }

      const updateData = {
        video_id: videoId,
        frame_idx: currentFrameIndex,
        obj_id: selectedObjectId,
        effects: effectData.effects || [],
        feather_params: effectData.feather_params || {},
        preview_mode: effectPreviewMode,
        reset_frame: true,
        original_frame_hash: originalFramesRef.current[frameKey]
      };

      const stateHash = JSON.stringify({
        effects: updateData.effects,
        feather_params: updateData.feather_params,
        frame_idx: updateData.frame_idx,
        preview_mode: updateData.preview_mode
      });

      if (lastAppliedStateRef.current === stateHash) {
        processingFramesRef.current.delete(frameKey);
        return;
      }

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

  // Handle frame changes
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

  // Handle initial frame
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

  // Cleanup with proper ref capturing
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