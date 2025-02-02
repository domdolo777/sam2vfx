import { useState, useCallback, useRef } from 'react';

export const useEffectState = (initialObjects = [], onEffectUpdate) => {
  const [objects, setObjects] = useState(initialObjects);
  const previousStateRef = useRef({});
  const updateQueueRef = useRef([]);

  // Create state update function
  const performStateUpdate = useCallback((newState) => {
    setObjects(newState);
    if (onEffectUpdate) {
      onEffectUpdate();
    }
  }, [onEffectUpdate]);

  // Debounced update handler
  const handleStateUpdate = useCallback((newState) => {
    window.requestAnimationFrame(() => {
      performStateUpdate(newState);
    });
  }, [performStateUpdate]);

  // Update effect parameters
  const updateEffectParameter = useCallback((objectId, effectIndex, paramName, value) => {
    const stateKey = `${objectId}-${effectIndex}-${paramName}`;
    const targetObject = objects.find(obj => obj.id === objectId);
    
    if (!targetObject?.effects?.[effectIndex]) return;

    previousStateRef.current[stateKey] = targetObject.effects[effectIndex].params[paramName];

    const updatedObjects = objects.map(obj => {
      if (obj.id === objectId) {
        const updatedEffects = [...obj.effects];
        updatedEffects[effectIndex] = {
          ...updatedEffects[effectIndex],
          params: {
            ...updatedEffects[effectIndex].params,
            [paramName]: value
          }
        };
        return { ...obj, effects: updatedEffects };
      }
      return obj;
    });

    updateQueueRef.current.push({
      type: 'parameter',
      objectId,
      effectIndex,
      paramName,
      value
    });

    handleStateUpdate(updatedObjects);
  }, [objects, handleStateUpdate]);

  // Update effect properties
  const updateEffectProperty = useCallback((objectId, effectIndex, propertyName, value) => {
    const stateKey = `${objectId}-${effectIndex}-${propertyName}`;
    const targetObject = objects.find(obj => obj.id === objectId);
    
    if (!targetObject?.effects?.[effectIndex]) return;

    previousStateRef.current[stateKey] = targetObject.effects[effectIndex][propertyName];

    const updatedObjects = objects.map(obj => {
      if (obj.id === objectId) {
        const updatedEffects = [...obj.effects];
        updatedEffects[effectIndex] = {
          ...updatedEffects[effectIndex],
          [propertyName]: value
        };
        return { ...obj, effects: updatedEffects };
      }
      return obj;
    });

    updateQueueRef.current.push({
      type: 'property',
      objectId,
      effectIndex,
      propertyName,
      value
    });

    handleStateUpdate(updatedObjects);
  }, [objects, handleStateUpdate]);

  // Update feather parameters
  const updateFeatherParams = useCallback((objectId, params) => {
    const updatedObjects = objects.map(obj => {
      if (obj.id === objectId) {
        return {
          ...obj,
          featherParams: {
            ...obj.featherParams,
            ...params
          }
        };
      }
      return obj;
    });

    updateQueueRef.current.push({
      type: 'feather',
      objectId,
      params
    });

    handleStateUpdate(updatedObjects);
  }, [objects, handleStateUpdate]);

  // Revert last change
  const revertLastChange = useCallback(() => {
    if (updateQueueRef.current.length === 0) return;

    const lastUpdate = updateQueueRef.current[updateQueueRef.current.length - 1];
    const { objectId, effectIndex } = lastUpdate;

    if (lastUpdate.type === 'parameter') {
      const { paramName } = lastUpdate;
      const previousValue = previousStateRef.current[`${objectId}-${effectIndex}-${paramName}`];
      if (previousValue !== undefined) {
        updateEffectParameter(objectId, effectIndex, paramName, previousValue);
      }
    } else if (lastUpdate.type === 'property') {
      const { propertyName } = lastUpdate;
      const previousValue = previousStateRef.current[`${objectId}-${effectIndex}-${propertyName}`];
      if (previousValue !== undefined) {
        updateEffectProperty(objectId, effectIndex, propertyName, previousValue);
      }
    }

    updateQueueRef.current.pop();
  }, [updateEffectParameter, updateEffectProperty]);

  return {
    objects,
    setObjects: performStateUpdate,
    updateEffectParameter,
    updateEffectProperty,
    updateFeatherParams,
    revertLastChange
  };
};

export default useEffectState;