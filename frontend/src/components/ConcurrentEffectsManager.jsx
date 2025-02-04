import { useEffect, useCallback, useRef, useState } from 'react';
import axios from 'axios';
import md5 from 'crypto-js/md5';
import CryptoJS from 'crypto-js';

const api = axios.create();

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
  const processingFramesRef = useRef(new Set());
  const initialRenderRef = useRef(true);

  const generateFrameHash = async (frameIdx) => {
    try {
      const paddedIndex = String(frameIdx).padStart(5, '0');
      const frameUrl = `/videos/${videoId}/frames/${paddedIndex}.jpg?v=${Date.now()}`;
      const response = await fetch(frameUrl);
      const buffer = await response.arrayBuffer();
      return md5(CryptoJS.lib.WordArray.create(new Uint8Array(buffer))).toString();
    } catch (error) {
      console.error('Frame hash generation failed:', error);
      return Date.now().toString();
    }
  };

    const handleParameterUpdate = useCallback(async (effectData) => {
      if (isProcessing) return;
    
      try {
        setIsProcessing(true);
        const frameHash = await generateFrameHash(currentFrameIndex);
    
        // Ensure effects array exists and feather_params has required structure
        const updateData = {
          video_id: videoId,
          frame_idx: currentFrameIndex,
          obj_id: selectedObjectId,
          effects: effectData.effects || [],
          feather_params: {
            radius: effectData.feather_params?.radius || 0,
            expand: effectData.feather_params?.expand || 0,
            opacity: effectData.feather_params?.opacity || 1,
            invert_mask: effectData.feather_params?.invert_mask || false,
            invert_intensity: effectData.feather_params?.invert_intensity || 1
          },
          preview_mode: effectPreviewMode,
          reset_frame: true,
          original_frame_hash: frameHash
        };

      const stateHash = JSON.stringify({
        effects: updateData.effects,
        feather_params: updateData.feather_params,
        frame_idx: updateData.frame_idx
      });

      if (lastAppliedStateRef.current === stateHash) return;

      const response = await api.post('/apply_effects', updateData);

      if (response.status === 200) {
        lastAppliedStateRef.current = stateHash;
        onEffectUpdate?.();
      }
    } catch (error) {
      if (error.response) {
        if (error.response.status === 422) {
          const errorMsg = error.response.data.detail?.[0]?.msg || 'Invalid parameters';
          console.error('Validation Error:', errorMsg);
          alert(`Validation Error: ${errorMsg}`);
        }
        if (error.response.status === 409) {
          alert('Frame has changed - please reselect object');
        }
      }
      console.error('Effect application error:', error);
    } finally {
      setIsProcessing(false);
    }
  }, [videoId, currentFrameIndex, selectedObjectId, effectPreviewMode, onEffectUpdate, isProcessing]);

  useEffect(() => {
    lastAppliedStateRef.current = null;
    if (effectsMode && selectedObjectId) {
      const selectedObject = objects.find(obj => obj.id === selectedObjectId);
      if (selectedObject) {
        handleParameterUpdate({
          effects: selectedObject.effects,
          feather_params: selectedObject.featherParams
        });
      }
    }
  }, [currentFrameIndex, effectsMode, objects, selectedObjectId, handleParameterUpdate]);

  useEffect(() => {
    if (!selectedObjectId || !initialRenderRef.current) return;
    initialRenderRef.current = false;
    const timeout = setTimeout(() => {
      const selectedObject = objects.find(obj => obj.id === selectedObjectId);
      if (selectedObject) {
        handleParameterUpdate({
          effects: selectedObject.effects,
          feather_params: selectedObject.featherParams
        });
      }
    }, 500);
    return () => clearTimeout(timeout);
  }, [objects, selectedObjectId, handleParameterUpdate]);

  useEffect(() => {
    return () => {
      if (pendingRequestRef.current?.cancel) {
        pendingRequestRef.current.cancel();
      }
    };
  }, []);

  return null;
};

export default ConcurrentEffectsManager;
