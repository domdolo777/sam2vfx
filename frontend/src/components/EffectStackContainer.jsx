// EffectStacksContainer.jsx
import React, { useMemo } from 'react';
import EffectBlock from './EffectBlock';

const EffectStacksContainer = ({
  selectedObjectId,
  objects,
  effectLibrary,
  defaultParamConfigs,
  colorPresets,
  moveEffect,
  deleteEffect,
  toggleMuteEffect,
  handleEffectParameterChange,
  handleEffectPropertyChange,
  saveColorPreset
}) => {
  // Get the selected object's effects safely
  const selectedObject = useMemo(() => 
    objects?.find(obj => obj.id === selectedObjectId) || null
  , [objects, selectedObjectId]);

  const effects = useMemo(() => 
    selectedObject?.effects || []
  , [selectedObject]);

  // Early return if no selected object or effects
  if (!selectedObject || !effects) {
    return <div className="mt-4">No effects available</div>;
  }

  return (
    <div className="effect-stacks-container mt-4">
      {effects.map((effect, index) => (
        <EffectBlock
          key={`${effect.name}-${index}`}
          effect={effect}
          index={index}
          moveEffect={moveEffect}
          deleteEffect={deleteEffect}
          toggleMuteEffect={toggleMuteEffect}
          handleEffectParameterChange={handleEffectParameterChange}
          handleEffectPropertyChange={handleEffectPropertyChange}
          effectLibrary={effectLibrary || []}
          defaultParamConfigs={defaultParamConfigs || {}}
          colorPresets={colorPresets || []}
          saveColorPreset={saveColorPreset}
        />
      ))}
    </div>
  );
};

export default EffectStacksContainer;
