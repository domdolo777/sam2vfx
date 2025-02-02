import { useState, useCallback } from 'react';
import axios from 'axios';

export const useObjectCleanup = ({ videoId, currentFrameIndex, setObjects, setSelectedObjectId, setFrameVersion, setMarkers, setPromptsByObject }) => {
  const [isDeleting, setIsDeleting] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  const handleObjectDeletion = useCallback(async (id, selectedObjectId) => {
    if (isDeleting) return;

    try {
      setIsDeleting(true);
      setErrorMessage('');

      // Clear effects for this object
      await axios.post('/apply_effects', {
        video_id: videoId,
        frame_idx: currentFrameIndex,
        obj_id: id,
        effects: [],
        feather_params: {},
      });

      // Delete object from backend
      await axios.post('/delete_object', {
        video_id: videoId,
        obj_id: id,
      });

      // Update local state
      setObjects(prevObjects => {
        const newObjects = prevObjects.filter(obj => obj.id !== id);
        
        if (selectedObjectId === id) {
          if (newObjects.length > 0) {
            setSelectedObjectId(newObjects[0].id);
          } else {
            setSelectedObjectId(null);
          }
        }
        
        return newObjects;
      });

      // Clean up related state
      setMarkers(prev => prev.filter(marker => marker.objId !== id));
      setPromptsByObject(prev => {
        const updatedPrompts = { ...prev };
        delete updatedPrompts[id];
        return updatedPrompts;
      });

      // Force frame refresh
      setFrameVersion(prev => prev + 1);

    } catch (error) {
      console.error('Error deleting object:', error);
      setErrorMessage('Failed to delete object. Please try again.');
    } finally {
      setIsDeleting(false);
    }
  }, [
    videoId,
    currentFrameIndex,
    setObjects,
    setSelectedObjectId,
    setFrameVersion,
    setMarkers,
    setPromptsByObject,
    isDeleting
  ]);

  return {
    handleObjectDeletion,
    isDeleting,
    errorMessage
  };
};