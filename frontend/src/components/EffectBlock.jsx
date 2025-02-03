// EffectBlock.jsx
import React, { useRef, useState, useCallback, useMemo } from 'react';
import { useDrag, useDrop } from 'react-dnd';
import Slider from '@mui/material/Slider';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faChevronDown,
  faChevronUp,
  faChevronLeft,
  faChevronRight,
} from '@fortawesome/free-solid-svg-icons';
import { SketchPicker } from 'react-color';

const blendingModes = [
  'Normal',
  'Multiply',
  'Screen',
  'Overlay',
  'Darken',
  'Lighten',
  'Color Dodge',
  'Color Burn',
  'Hue',
  'Saturation',
  'Color',
  'Luminosity',
];


const EffectBlock = ({
  effect,
  index,
  moveEffect,
  deleteEffect,
  toggleMuteEffect,
  handleEffectParameterChange,
  handleEffectPropertyChange,
  effectLibrary = [],
  defaultParamConfigs = {},
  colorPresets = [],
  saveColorPreset,
}) => {
  const ref = useRef(null);
  const [isColorPickerOpen, setIsColorPickerOpen] = useState(false);
  const [selectedColorComponent, setSelectedColorComponent] = useState('backgroundColor');
  const [isBlendingModeMenuOpen, setIsBlendingModeMenuOpen] = useState(false);
  const [isPresetMenuOpen, setIsPresetMenuOpen] = useState(false);

  // Memoized values
  const colorSettings = useMemo(() => effect?.colorSettings || {}, [effect]);
  const effectName = useMemo(() => effect?.name || 'Unnamed Effect', [effect]);
  const effectParams = useMemo(() => effect?.params || {}, [effect]);
  const isMuted = useMemo(() => effect?.muted || false, [effect]);
  const isCollapsed = useMemo(() => effect?.collapsed || false, [effect]);
  const blendMode = useMemo(() => effect?.blend_mode || 'Normal', [effect]);

  const effectBlockStyle = useMemo(() => ({
    backgroundColor: colorSettings.backgroundColor || 'var(--effectBlockBg)',
    color: colorSettings.textColor || 'var(--effectBlockText)',
    border: '1px solid #ccc',
    borderRadius: '8px',
  }), [colorSettings]);

  // DnD setup
  const [, drop] = useDrop({
    accept: 'effect',
    hover(item) {
      if (!ref.current) return;
      const dragIndex = item.index;
      const hoverIndex = index;
      if (dragIndex === hoverIndex) return;
      moveEffect(dragIndex, hoverIndex);
      item.index = hoverIndex;
    },
  });

  const [{ isDragging }, drag] = useDrag({
    type: 'effect',
    item: { type: 'effect', index },
    collect: (monitor) => ({
      isDragging: monitor.isDragging(),
    }),
  });

  drag(drop(ref));

  // Callbacks
  const handleBlendModeChange = useCallback((direction) => {
    const currentIndex = blendingModes.indexOf(blendMode);
    const newIndex = direction === 'prev' 
      ? (currentIndex - 1 + blendingModes.length) % blendingModes.length
      : (currentIndex + 1) % blendingModes.length;
    handleEffectPropertyChange(index, 'blend_mode', blendingModes[newIndex]);
  }, [blendMode, handleEffectPropertyChange, index]);

  const renderParameterControl = useCallback((paramName, paramValue, paramConfig) => {
    return (
      <Slider
        value={paramValue}
        onChange={(e, value) => handleEffectParameterChange(index, paramName, value)}
        min={paramConfig.min}
        max={paramConfig.max}
        step={paramConfig.step || (paramConfig.type === 'int' ? 1 : 0.1)}
        className="mr-2 flex-1"
        sx={{
          color: colorSettings.sliderColor || 'var(--sliderColor)',
        }}
      />
    );
  }, [handleEffectParameterChange, index, colorSettings]);

  // Render feathering controls
  const renderFeatheringControls = useCallback(() => {
    const featherParams = {
      radius: effect.radius || 0,
      expand: effect.expand || 0,
      opacity: effect.opacity || 1.0
    };
  
    return (
      <div className="mt-4 border-t border-gray-300 pt-4">
        <h4 className="font-medium mb-2">Effect Feathering</h4>
        {Object.entries(featherParams).map(([param, value]) => (
          <div key={param} className="mt-2">
            <label className="block text-sm font-medium mb-1">
              {param.charAt(0).toUpperCase() + param.slice(1)}: {value}
            </label>
            <div className="flex items-center">
              <Slider
                value={value}
                onChange={(e, newValue) => {
                  handleEffectPropertyChange(index, param, newValue);
                }}
                min={param === 'expand' ? -20 : 0}
                max={param === 'expand' ? 20 : param === 'opacity' ? 1 : 100}
                step={param === 'opacity' ? 0.1 : 1}
                className="mr-2 flex-1"
                sx={{
                  color: colorSettings.sliderColor || 'var(--sliderColor)'
                }}
              />
              <input
                type="number"
                value={value}
                onChange={(e) => handleEffectPropertyChange(index, param, parseFloat(e.target.value))}
                min={param === 'expand' ? -20 : 0}
                max={param === 'expand' ? 20 : param === 'opacity' ? 1 : 100}
                step={param === 'opacity' ? 0.1 : 1}
                className="w-16 ml-2 border rounded px-2 py-1"
                style={{
                  backgroundColor: colorSettings.inputBackgroundColor || '#f0f0f0',
                  color: colorSettings.textColor || '#000000'
                }}
              />
            </div>
          </div>
        ))}
      </div>
    );
  }, [effect, index, handleEffectPropertyChange, colorSettings]);
  
  return (
    <div
      ref={ref}
      className={`p-2 mb-2 ${isDragging ? 'opacity-50' : 'opacity-100'} ${isMuted ? 'opacity-50' : ''}`}
      style={effectBlockStyle}
    >
      <div className="flex justify-between items-center">
        <button
          onClick={() => handleEffectPropertyChange(index, 'collapsed', !isCollapsed)}
          className="mr-2"
        >
          <FontAwesomeIcon icon={isCollapsed ? faChevronDown : faChevronUp} />
        </button>

        <div className="flex items-center">
          <span className="font-semibold">{effectName}</span>
          <button
            onClick={() => setIsColorPickerOpen(!isColorPickerOpen)}
            className="ml-2"
            style={{
              backgroundColor: colorSettings.sliderColor || 'var(--sliderColor)',
              width: '20px',
              height: '20px',
              borderRadius: '50%',
            }}
          />
        </div>

        <div className="flex items-center">
          <div className="flex items-center mr-2 relative">
            <button onClick={() => handleBlendModeChange('prev')}>
              <FontAwesomeIcon icon={faChevronLeft} />
            </button>
            
            <button
              onClick={() => setIsBlendingModeMenuOpen(!isBlendingModeMenuOpen)}
              className="mx-1"
            >
              {blendMode}
            </button>

            <button onClick={() => handleBlendModeChange('next')}>
              <FontAwesomeIcon icon={faChevronRight} />
            </button>

            {isBlendingModeMenuOpen && (
              <div className="absolute bg-white border mt-2 z-10 max-h-40 overflow-y-auto">
                {blendingModes.map((mode) => (
                  <div
                    key={mode}
                    onClick={() => {
                      handleEffectPropertyChange(index, 'blend_mode', mode);
                      setIsBlendingModeMenuOpen(false);
                    }}
                    className="px-2 py-1 hover:bg-gray-200 cursor-pointer"
                  >
                    {mode}
                  </div>
                ))}
              </div>
            )}
          </div>

          <button
            onClick={() => toggleMuteEffect(index)}
            className="text-black hover:text-gray-700 mr-2"
          >
            {isMuted ? 'Unmute' : 'Mute'}
          </button>
          <button
            onClick={() => deleteEffect(index)}
            className="text-red-500 hover:text-red-700"
            style={{ color: colorSettings.deleteButtonColor || '#ff0000' }}
          >
            Delete
          </button>
        </div>
      </div>

      {isColorPickerOpen && (
        <div className="mt-2 p-2 border rounded bg-white">
          <div className="flex justify-between items-center">
            <span className="font-semibold">{effectName} - Color Settings</span>
            <button
              onClick={() => setIsColorPickerOpen(false)}
              className="text-sm text-blue-500"
            >
              Close
            </button>
          </div>

          <div className="mt-2">
            <label className="block text-sm font-medium">Select Component to Color:</label>
            <select
              value={selectedColorComponent}
              onChange={(e) => setSelectedColorComponent(e.target.value)}
              className="w-full mt-1 border rounded px-2 py-1"
            >
              <option value="backgroundColor">Effect Block Background</option>
              <option value="textColor">Text and Numbers</option>
              <option value="deleteButtonColor">Delete Button</option>
              <option value="inputBackgroundColor">Input Field Background</option>
              <option value="sliderColor">Sliders</option>
            </select>
          </div>

          <div className="mt-4">
            <SketchPicker
              color={colorSettings[selectedColorComponent] || '#ffffff'}
              onChangeComplete={(color) => {
                handleEffectPropertyChange(index, 'colorSettings', {
                  ...colorSettings,
                  [selectedColorComponent]: color.hex,
                });
              }}
            />
          </div>

          <button
            onClick={() => saveColorPreset(colorSettings)}
            className="mt-2 bg-green-500 text-white py-1 px-4 rounded hover:bg-green-600"
          >
            Save Color Preset
          </button>

          <button
            onClick={() => setIsPresetMenuOpen(!isPresetMenuOpen)}
            className="mt-2 bg-blue-500 text-white py-1 px-4 rounded hover:bg-blue-600"
          >
            Load Color Preset
          </button>

          {isPresetMenuOpen && (
            <div className="mt-2 bg-white border max-h-40 overflow-y-auto">
              {colorPresets.map((preset) => (
                <div
                  key={preset.preset_name}
                  onClick={() => {
                    handleEffectPropertyChange(index, 'colorSettings', {
                      ...colorSettings,
                      ...preset.color_settings,
                    });
                    setIsPresetMenuOpen(false);
                  }}
                  className="px-2 py-1 hover:bg-gray-200 cursor-pointer"
                  style={{
                    backgroundColor: preset.color_settings.backgroundColor || 'var(--effectBlockBg)',
                    color: preset.color_settings.textColor || 'var(--effectBlockText)',
                  }}
                >
                  {preset.preset_name}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {!isMuted && !isCollapsed && effectParams && Object.keys(effectParams).map((param) => {
        const paramConfig = effectLibrary.find((e) => e.name === effectName)?.defaultParams[param] || 
                          defaultParamConfigs[param];
        if (!paramConfig) return null;

        const paramType = paramConfig.type;
        const paramLabel = paramConfig.label || param.charAt(0).toUpperCase() + param.slice(1);
        const paramValue = effectParams[param];

        return (
          <div key={param} className="mt-2">
            <label
              className="block text-sm font-medium"
              style={{ color: colorSettings.textColor || '#000000' }}
            >
              {paramLabel}: {paramValue}
            </label>
            <div className="flex items-center">
              {renderParameterControl(param, paramValue, paramConfig)}
              <input
                type="number"
                value={paramValue}
                onChange={(e) => handleEffectParameterChange(index, param, e.target.value)}
                min={paramConfig.min}
                max={paramConfig.max}
                step={paramConfig.step || (paramType === 'int' ? 1 : 0.1)}
                className="w-16 ml-2 border rounded px-2 py-1"
                style={{
                  backgroundColor: colorSettings.inputBackgroundColor || '#f0f0f0',
                  color: colorSettings.textColor || '#000000',
                }}
              />
            </div>
          </div>
        );
      })}

      {!isMuted && !isCollapsed && renderFeatheringControls()}
    </div>
  );
};

export default EffectBlock;
