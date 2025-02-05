import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import Slider from '@mui/material/Slider';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import 'tailwindcss/tailwind.css';
import Modal from '@mui/material/Modal';
import Box from '@mui/material/Box';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faLink } from '@fortawesome/free-solid-svg-icons';
import { SketchPicker } from 'react-color'; // For color picker
import EffectBlock from './EffectBlock'; // Import your EffectBlock component
import ConcurrentEffectsManager from './ConcurrentEffectsManager';

function VideoEditor({ videoId }) {
   // State Hooks
  const [frames, setFrames] = useState([]);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [markers, setMarkers] = useState([]);
  const [objects, setObjects] = useState([]);
  const [selectedObjectId, setSelectedObjectId] = useState(null);
  const [effectsMode, setEffectsMode] = useState(false);
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });
  const [frameVersion, setFrameVersion] = useState(0);
  const [promptsByObject, setPromptsByObject] = useState({});
  const [promptMode, setPromptMode] = useState('point');
  const [promptType, setPromptType] = useState('positive');
  const [trackingInProgress, setTrackingInProgress] = useState(false);
  const [boxStart, setBoxStart] = useState(null);
  const [effectLibrary, setEffectLibrary] = useState([]);
  const [selectedEffectName, setSelectedEffectName] = useState('');
  const [isLoadingEffects, setIsLoadingEffects] = useState(false);
  const [effectStackPresets, setEffectStackPresets] = useState([]);
  const [isChainMode, setIsChainMode] = useState(false);
  const [isLoadPresetOpen, setIsLoadPresetOpen] = useState(false);
  const [effectPreviewMode, setEffectPreviewMode] = useState('all');
  const [concurrentEffectsEnabled, setConcurrentEffectsEnabled] = useState(true);

  // Import Effect Modal State
  const [importModalOpen, setImportModalOpen] = useState(false);
  const [newEffectName, setNewEffectName] = useState('');
  const [newEffectCode, setNewEffectCode] = useState('');
  const [newEffectConfig, setNewEffectConfig] = useState('');

  // State for Color Presets
  const [colorPresets, setColorPresets] = useState([]);
  const [isLoadColorPresetOpen, setIsLoadColorPresetOpen] = useState(false);
  const [newColorPresetName, setNewColorPresetName] = useState('');

  // State Hooks for Global Color Settings
  const [isGlobalColorModalOpen, setIsGlobalColorModalOpen] = useState(false);
  const [globalColorSettings, setGlobalColorSettings] = useState({});

  const defaultParamConfigs = {
    opacity: {
      label: 'Opacity',
      type: 'float',
      min: 0.0,
      max: 1.0,
      step: 0.1,
    },
    radius: {
      label: 'Radius',
      type: 'int',
      min: 0,
      max: 100,
      step: 1,
    },
    expand: {
      label: 'Expand',
      type: 'int',
      min: -50,
      max: 50,
      step: 1,
    },
  };

  const colorCategories = [
    {
      categoryName: 'Containers',
      components: [
        { name: 'Sidebar Background', key: 'sidebarBg', type: 'background' },
        { name: 'Main Background', key: 'mainBg', type: 'background' },
        // Add more containers as needed
      ],
    },
    {
      categoryName: 'Texts',
      components: [
        { name: 'Main Text', key: 'mainText', type: 'text' },
        // Add more texts as needed
      ],
    },
    {
      categoryName: 'Buttons',
      components: [
        { name: 'Primary Button Background', key: 'primaryButtonBg', type: 'background' },
        { name: 'Primary Button Text', key: 'primaryButtonText', type: 'text' },
        { name: 'Primary Button Hover Background', key: 'primaryButtonHoverBg', type: 'background' },
        { name: 'Disabled Button Background', key: 'disabledButtonBg', type: 'background' },
        { name: 'Disabled Button Text', key: 'disabledButtonText', type: 'text' },
        // Add more buttons as needed
      ],
    },
    {
      categoryName: 'Effect Blocks',
      components: [
        { name: 'Effect Block Background', key: 'effectBlockBg', type: 'background' },
        { name: 'Effect Block Text', key: 'effectBlockText', type: 'text' },
        // Add more effect block components as needed
      ],
    },
    {
      categoryName: 'Prompt Buttons',
      components: [
        { name: 'Button Background Selected', key: 'buttonBgSelected', type: 'background' },
        { name: 'Button Text Selected', key: 'buttonTextSelected', type: 'text' },
        { name: 'Button Background Unselected', key: 'buttonBgUnselected', type: 'background' },
        { name: 'Button Text Unselected', key: 'buttonTextUnselected', type: 'text' },
        { name: 'Button Background Hover', key: 'buttonBgHover', type: 'background' },
      ],
    },
    {
      categoryName: 'Sliders',
      components: [
        { name: 'Slider Color', key: 'sliderColor', type: 'color' },
      ],
    },
    // Add other categories as needed
  ];

  // Apply Global Colors
  const applyGlobalColors = useCallback((colorSettings) => {
    if (!colorSettings) return;

    Object.keys(colorSettings).forEach((key) => {
      const cssVariableName = `--${key}`;
      document.documentElement.style.setProperty(cssVariableName, colorSettings[key]);
    });

    // Handle effect block colors
    const effectBlockBg = colorSettings.effectBlockBg || '#ffffff';
    const effectBlockText = colorSettings.effectBlockText || '#000000';

    setObjects((prevObjects) =>
      prevObjects.map((obj) => {
        const updatedEffects = obj.effects.map((effect) => ({
          ...effect,
          colorSettings: {
            backgroundColor: effectBlockBg,
            textColor: effectBlockText,
          },
        }));
        return { ...obj, effects: updatedEffects };
      })
    );
  }, []);

  // Function to load the last used color preset
  const loadLastUsedColorPreset = useCallback(async () => {
    try {
      const response = await axios.get('/get_last_used_color_preset');
      if (response.status === 200 && response.data) {
        setGlobalColorSettings(response.data.color_settings);
        applyGlobalColors(response.data.color_settings);
      } else {
        // If no last used preset, apply default colors
        applyGlobalColors({});
      }
    } catch (error) {
      console.error('Failed to load last used color preset:', error);
      // Apply default colors in case of error
      applyGlobalColors({});
    }
  }, [applyGlobalColors]);

  // Fetch Color Presets
  const fetchColorPresets = useCallback(async () => {
    try {
      const response = await axios.get('/get_color_presets');
      setColorPresets(response.data); // Assuming the API returns an array of presets
    } catch (error) {
      console.error('Error fetching color presets:', error);
    }
  }, []);

  // Fetch Effect Library
  const fetchEffectLibrary = useCallback(async () => {
    setIsLoadingEffects(true);
    try {
      const response = await axios.get(`/get_effects/${videoId}`);
      setEffectLibrary(response.data.effects);
    } catch (error) {
      console.error('Error fetching effect library:', error);
    } finally {
      setIsLoadingEffects(false);
    }
  }, [videoId]);

  // Load Color Presets
  const loadColorPresets = useCallback(async () => {
    try {
      const response = await axios.get('/load_color_presets');
      setColorPresets(response.data.presets);
    } catch (error) {
      console.error('Error loading color presets:', error);
    }
  }, []);

  // Load Effect Stack Presets
  const loadEffectStackPresets = useCallback(async () => {
    try {
      const response = await axios.get('/load_effect_stack_presets');
      setEffectStackPresets(response.data.presets);
    } catch (error) {
      console.error('Error loading effect stack presets:', error);
    }
  }, []);

  // Fetch Effect Library and Color Presets
  useEffect(() => {
    fetchEffectLibrary();
    fetchColorPresets();
  }, [fetchEffectLibrary, fetchColorPresets]);

  // Load Presets on Mount
  useEffect(() => {
    loadColorPresets();
    loadEffectStackPresets();
    loadLastUsedColorPreset(); // Load last used color preset on mount
  }, [loadColorPresets, loadEffectStackPresets, loadLastUsedColorPreset]);

  // Function to save the current color settings as a preset
  const saveColorSettingsPreset = async () => {
    if (!newColorPresetName) {
      alert('Please enter a name for the preset.');
      return;
    }

    const presetData = {
      preset_name: newColorPresetName,
      color_settings: globalColorSettings,
    };

    try {
      const response = await axios.post('/save_color_preset', presetData);
      if (response.status === 200) {
        alert('Color preset saved successfully.');
        setNewColorPresetName('');
        fetchColorPresets(); // Refresh the list of presets
      }
    } catch (error) {
      console.error('Failed to save color preset:', error);
      alert('Failed to save color preset. Please try again.');
    }
  };

  // Function to load a selected color preset
  const loadColorSettingsPreset = (preset) => {
    if (!preset || !preset.color_settings) {
      alert('Invalid preset.');
      return;
    }

    // Merge the preset color settings with defaults
    const defaultColorSettings = {
      sidebarBg: '#12190f',
      mainBg: '#3c343b',
      mainText: '#e6e6e6',
      effectBlockBg: '#ffffff',
      effectBlockText: '#000000',
      primaryButtonBg: '#aa797d',
      primaryButtonText: '#ffffff',
      primaryButtonHoverBg: '#9a6a6e',
      disabledButtonBg: '#cccccc',
      disabledButtonText: '#666666',
      buttonBgSelected: '#128eb8',
      buttonTextSelected: '#ffffff',
      buttonBgUnselected: '#e5e7eb',
      buttonTextUnselected: '#1f2937',
      buttonBgHover: '#d1d5db',
      sliderColor: '#9eaae7', // Default slider color
      // Add any other default color settings
    };

    const mergedColorSettings = { ...defaultColorSettings, ...preset.color_settings };

    setGlobalColorSettings(mergedColorSettings);
    applyGlobalColors(mergedColorSettings);
    setIsLoadColorPresetOpen(false);
  };

  // Function to open the Import Modal
  const handleOpenImportModal = () => {
    setImportModalOpen(true);
  };

  // Function to close the Import Modal
  const handleCloseImportModal = () => {
    setImportModalOpen(false);
  };

  // Function to handle importing the effect
  const handleImportEffect = async () => {
    const effectData = {
      video_id: videoId,
      effect_name: newEffectName,
      effect_code: newEffectCode,
      effect_config: newEffectConfig || null,
    };

    try {
      const response = await axios.post('/upload_effect', effectData);
      if (response.status === 200) {
        // Effect uploaded successfully, now fetch the updated effect list
        await fetchEffectLibrary();
        // Reset modal state
        setImportModalOpen(false);
        setNewEffectName('');
        setNewEffectCode('');
        setNewEffectConfig('');
      }
    } catch (error) {
      console.error('Failed to upload effect:', error);
      alert('Failed to upload effect. Please try again.');
    }
  };

  const saveColorPreset = async (colorSettings) => {
    const presetName = prompt('Enter a name for the color preset:');
    if (!presetName) return;

    try {
      await axios.post('/save_color_preset', {
        preset_name: presetName,
        color_settings: colorSettings,
      });
      alert('Color preset saved successfully.');
      fetchColorPresets(); // Refresh the list of presets
    } catch (error) {
      console.error('Error saving color preset:', error);
      alert('Failed to save color preset.');
    }
  };

  // Function to save effect stack preset
  const saveEffectStackPreset = async () => {
    const presetName = prompt('Enter a name for the effect stack preset:');
    if (!presetName) return;

    const subFolder = prompt('Enter a subfolder (optional):') || '';

    const selectedObject = objects.find((o) => o.id === selectedObjectId);
    if (!selectedObject) return;

    try {
      await axios.post('/save_effect_stack_preset', {
        preset_name: presetName,
        effects_stack: selectedObject.effects,
        sub_folder: subFolder,
      });
      alert('Effect stack preset saved successfully.');
      loadEffectStackPresets(); // Reload presets after saving
    } catch (error) {
      console.error('Error saving effect stack preset:', error);
      alert('Failed to save effect stack preset.');
    }
  };

  // Function to load effect stack
  const loadEffectStack = (effectsStack) => {
    setObjects((prevObjects) =>
      prevObjects.map((obj) => {
        if (obj.id === selectedObjectId) {
          const updatedEffects = isChainMode ? [...obj.effects, ...effectsStack] : effectsStack;
          return { ...obj, effects: updatedEffects };
        }
        return obj;
      })
    );
    // Apply effects after loading
    if (effectsMode) {
      applyEffects();
    }
  };

  // Handle Effect Property Changes (e.g., collapsed, blend_mode, colorSettings)
  const handleEffectPropertyChange = (effectIndex, property, value) => {
    setObjects((prevObjects) =>
      prevObjects.map((obj) => {
        if (obj.id === selectedObjectId) {
          const updatedEffects = obj.effects.map((eff, idx) => {
            if (idx === effectIndex) {
              return {
                ...eff,
                [property]: value,
              };
            }
            return eff;
          });
          return { ...obj, effects: updatedEffects };
        }
        return obj;
      })
    );
    // Re-apply effects if necessary
    if (effectsMode && property !== 'collapsed' && property !== 'colorSettings') {
      applyEffects();
    }
  };

  // Enhanced State Variables for Invert Mask and Invert Intensity within featherParams
  const [featherParams, setFeatherParams] = useState({
    radius: 0,
    expand: 0,
    opacity: 1,
    invert_mask: false, // New Parameter
    invert_intensity: 1.0, // New Parameter
  });

  const canvasRef = useRef(null);
  const intervalRef = useRef(null);

  // Fetch Frames
  useEffect(() => {
    const fetchFrames = async () => {
      try {
        const response = await axios.get(`/get_frames/${videoId}`);
        setFrames(response.data.frames);
      } catch (error) {
        console.error('Error fetching frames:', error);
        alert('Failed to load video frames.');
      }
    };
    fetchFrames();
  }, [videoId]);

  // Initialize Objects
  useEffect(() => {
    if (objects.length === 0) {
      setObjects([
        {
          id: 1,
          name: 'Object 1',
          maskOpacity: 0.5,
          featherParams: {
            radius: 0,
            expand: 0,
            opacity: 1,
            invert_mask: false,
            invert_intensity: 1.0,
          }, // Initialize new params
          effects: [],
          muted: false,
        },
      ]);
      setSelectedObjectId(1);
    }
  }, [objects]);

  // Draw Masks
  const drawMasks = useCallback(
    (masks) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      Object.keys(masks).forEach((objId) => {
        const object = objects.find((obj) => obj.id === parseInt(objId));
        if (object && !object.muted) {
          const maskData = masks[objId];
          const img = new Image();
          img.src = `data:image/png;base64,${maskData}`;
          img.onload = () => {
            ctx.globalAlpha = object.maskOpacity;
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            ctx.globalAlpha = 1.0;
          };
        }
      });
    },
    [objects]
  );

  // Fetch Masks
  const fetchMasks = useCallback(
    async (frameIdx) => {
      try {
        const response = await axios.get(`/get_masks/${videoId}/${frameIdx}`);
        const masks = response.data.masks;
        drawMasks(masks);
      } catch (error) {
        console.error('Error fetching masks:', error);
      }
    },
    [videoId, drawMasks]
  );

  // Update Masks when Frame Changes
  useEffect(() => {
    if (frames.length > 0) {
      fetchMasks(currentFrameIndex);
      setFrameVersion((prev) => prev + 1); // Force frame image to reload
    }
  }, [currentFrameIndex, frames, fetchMasks]);

  // Handle Playing Video
  useEffect(() => {
    if (playing) {
      intervalRef.current = setInterval(() => {
        setCurrentFrameIndex((prevIndex) =>
          prevIndex < frames.length - 1 ? prevIndex + 1 : prevIndex
        );
      }, 1000 / 24);
    } else {
      clearInterval(intervalRef.current);
    }
    return () => clearInterval(intervalRef.current);
  }, [playing, frames]);

  // Handle Frame Click for Prompts
  const handleFrameClick = async (e) => {
    if (effectsMode) return;

    const imgElement = e.target;
    const rect = imgElement.getBoundingClientRect();
    const scaleX = imageDimensions.width / rect.width;
    const scaleY = imageDimensions.height / rect.height;

    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    if (promptMode === 'point') {
      setMarkers((prevMarkers) => [
        ...prevMarkers,
        { x, y, type: promptType, objId: selectedObjectId, frameIdx: currentFrameIndex },
      ]);

      setPromptsByObject((prev) => {
        const prevPrompts = prev[selectedObjectId] || { points: [], labels: [], box: null };
        return {
          ...prev,
          [selectedObjectId]: {
            ...prevPrompts,
            points: [...prevPrompts.points, [x, y]],
            labels: [...prevPrompts.labels, promptType === 'positive' ? 1 : 0],
          },
        };
      });
    } else if (promptMode === 'box') {
      if (!boxStart) {
        setBoxStart({ x, y });
      } else {
        const box = [boxStart.x, boxStart.y, x, y];
        setMarkers((prevMarkers) => [
          ...prevMarkers,
          { x: boxStart.x, y: boxStart.y, type: 'box_start', objId: selectedObjectId, frameIdx: currentFrameIndex },
          { x, y, type: 'box_end', objId: selectedObjectId, frameIdx: currentFrameIndex },
        ]);

        setPromptsByObject((prev) => ({
          ...prev,
          [selectedObjectId]: {
            ...prev[selectedObjectId],
            box: box,
          },
        }));

        setBoxStart(null);
      }
    }
  };

  // Send Prompts to Backend
  const sendBatchPrompts = async () => {
    const objectIds = Object.keys(promptsByObject);
    for (let objId of objectIds) {
      const prompts = promptsByObject[objId];
      const data = {
        video_id: videoId, // Use the videoId from props
        frame_idx: currentFrameIndex, // Current frame index from state
        obj_id: parseInt(objId), // Object ID as an integer
        points: prompts.points || [], // Points from promptsByObject
        labels: prompts.labels || [], // Labels (0 or 1)
        box: prompts.box || null, // Box if available
        clear_old_prompts: false, // Set based on the requirement
        normalize_coords: false, // Assuming you're not normalizing coordinates
      };

      try {
        // Send the data to the backend
        await axios.post('/add_prompts', data);
        console.log(`Prompts for object ${objId} sent successfully`);
      } catch (error) {
        console.error(`Error sending prompts for object ${objId}:`, error);
        alert(`Failed to apply prompts for object ${objects.find(obj => obj.id === parseInt(objId)).name}.`);
      }
    }

    // Optionally, fetch the updated masks after sending prompts
    fetchMasks(currentFrameIndex);
  };

// In VideoEditor.js

const handleSliderChange = (param, value) => {
  const parsedValue = parseFloat(value);
  
  // Update feather params
  setFeatherParams(prev => ({
    ...prev,
    [param]: parsedValue,
  }));

  // Update selected object
  setObjects(prevObjects =>
    prevObjects.map(obj =>
      obj.id === selectedObjectId
        ? {
            ...obj,
            featherParams: {
              ...obj.featherParams,
              [param]: parsedValue,
              // Ensure required fields are always present
              radius: obj.featherParams?.radius ?? 0,
              expand: obj.featherParams?.expand ?? 0,
              opacity: obj.featherParams?.opacity ?? 1.0
            },
          }
        : obj
    )
  );
};


  const handleCheckboxChange = (param, checked) => {
    setFeatherParams((prev) => ({
      ...prev,
      [param]: checked,
    }));
  
    // Update the selected object's featherParams
    setObjects((prevObjects) =>
      prevObjects.map((obj) =>
        obj.id === selectedObjectId
          ? {
              ...obj,
              featherParams: {
                ...obj.featherParams,
                [param]: checked,
              },
            }
          : obj
      )
    );
  };

  // Modified applyEffects Function to Include New Parameters within featherParams
// In VideoEditor.js

const applyEffects = useCallback(async () => {
  const selectedObject = objects.find((o) => o.id === selectedObjectId);
  if (!selectedObject) return;

  try {
    const effectData = {
      video_id: videoId,
      obj_id: selectedObjectId,
      effects: selectedObject.effects,
      feather_params: selectedObject.featherParams,
      frame_idx: currentFrameIndex,
      apply_to_all_frames: false,
      preview_mode: effectPreviewMode // Add this line
    };

    console.log('Sending apply effects request with data:', JSON.stringify(effectData));

    await axios.post('/apply_effects', effectData);
    setFrameVersion((prev) => prev + 1);
  } catch (error) {
    console.error('Error applying effects:', error);
    alert('Error applying effects.');
  }
}, [videoId, selectedObjectId, objects, currentFrameIndex, effectPreviewMode]); // Add effectPreviewMode to dependencies

// In VideoEditor.js

const applyEffectsToAllFrames = useCallback(async () => {
  const selectedObject = objects.find((o) => o.id === selectedObjectId);
  if (!selectedObject) return;

  try {
    const effectData = {
      video_id: videoId,
      obj_id: selectedObjectId,
      effects: selectedObject.effects,
      feather_params: selectedObject.featherParams, // Use selected object's featherParams
      apply_to_all_frames: true,
    };

    // Log request data for debugging
    console.log('Sending apply effects to all frames request with data:', JSON.stringify(effectData));

    // Make the request
    await axios.post('/apply_effects', effectData);
    setFrameVersion((prev) => prev + 1);
  } catch (error) {
    console.error('Error applying effects to all frames:', error);
    alert('Error applying effects to all frames.');
  }
}, [videoId, selectedObjectId, objects]);


  // Ensure the `addEffect` function uses the effectLibrary
  const addEffect = () => {
    if (!selectedEffectName) return;
    const effect = effectLibrary.find((e) => e.name === selectedEffectName);
    if (effect) {
      const initializedParams = effect.defaultParams
        ? Object.fromEntries(
            Object.entries(effect.defaultParams).map(([key, param]) => [key, param.value])
          )
        : {};
      // Add default opacity, radius, expand
      initializedParams['opacity'] = 1.0;
      initializedParams['radius'] = 0;
      initializedParams['expand'] = 0;

      setObjects((prevObjects) =>
        prevObjects.map((obj) =>
          obj.id === selectedObjectId
            ? {
                ...obj,
                effects: [
                  ...obj.effects,
                  {
                    name: selectedEffectName,
                    params: initializedParams,
                    muted: false,
                    colorSettings: {}, // Initialize colorSettings
                  },
                ],
              }
            : obj
        )
      );
      // Apply effects after adding new effect
      if (effectsMode) {
        applyEffects();
      }
      // Reset the select value
      setSelectedEffectName('');
    }
  };

  // Toggle Effects Mode
  const toggleEffectsMode = () => {
    setEffectsMode(!effectsMode);
    if (!effectsMode) {
      // Entering effects mode, apply effects
      applyEffects();
    }
  };

  // Track Objects
  const trackObjects = async () => {
    try {
      setTrackingInProgress(true);
      await axios.post('/track_objects', { video_id: videoId });
      const checkTrackingStatus = async () => {
        try {
          const statusResponse = await axios.get(`/tracking_status/${videoId}`);
          if (statusResponse.data.status === 'complete') {
            setTrackingInProgress(false);
            alert('Tracking complete.');
            setFrameVersion((prev) => prev + 1);
          } else {
            setTimeout(checkTrackingStatus, 2000);
          }
        } catch (error) {
          console.error('Error checking tracking status:', error);
          setTrackingInProgress(false);
        }
      };
      checkTrackingStatus();
    } catch (error) {
      console.error('Error tracking objects:', error);
      setTrackingInProgress(false);
      alert('Error initiating tracking.');
    }
  };

  // Export Video
  const exportVideo = useCallback(
    async (options) => {
      try {
        const response = await axios.post('/export', {
          video_id: videoId,
          export_options: options,
        });
        window.open(`${response.data.download_url}`, '_blank');
      } catch (error) {
        console.error('Error exporting video:', error);
        alert('Error exporting video.');
      }
    },
    [videoId]
  );

  // Reset Application State
  const resetState = useCallback(async () => {
    try {
      await axios.post('/reset_state', { video_id: videoId });
      setCurrentFrameIndex(0);
      setPlaying(false);
      setMarkers([]);
      setObjects([]);
      setSelectedObjectId(null);
      setEffectsMode(false);
      setFrameVersion((prev) => prev + 1);
      setPromptsByObject({});
      setEffectLibrary([]);
      setNewEffectName('');
      setNewEffectCode('');
      setNewEffectConfig('');
      setFeatherParams({
        radius: 0,
        expand: 0,
        opacity: 1,
        invert_mask: false,
        invert_intensity: 1.0,
      }); // Reset featherParams including new parameters
      alert('Application state has been reset.');
    } catch (error) {
      console.error('Error resetting state:', error);
      alert('Error resetting state.');
    }
  }, [videoId]);

  // Handle Play/Pause
  const togglePlay = () => {
    setPlaying(!playing);
  };

  // Frame Navigation
  const nextFrame = () => {
    setCurrentFrameIndex((prevIndex) =>
      prevIndex < frames.length - 1 ? prevIndex + 1 : prevIndex
    );
  };

  const prevFrame = () => {
    setCurrentFrameIndex((prevIndex) => (prevIndex > 0 ? prevIndex - 1 : prevIndex));
  };

  // Add New Object
// In VideoEditor.js

// Replace your existing addObject function with this one
  const addObject = () => {
    const newId = objects.length > 0 ? objects[objects.length - 1].id + 1 : 1;
    setObjects([
      ...objects,
      {
        id: newId,
        name: `Object ${newId}`,
        maskOpacity: 0.5,
        featherParams: {
          radius: 0,
          expand: 0,
          opacity: 1,
          invert_mask: false,
          invert_intensity: 1.0,
        },
        effects: [],
        muted: false,
      },
    ]);
    setSelectedObjectId(newId);
  };


  // Delete Object
  const deleteObject = async (id) => {
    try {
      await axios.post('/delete_object', {
        video_id: videoId,
        obj_id: id,
      });
      setObjects(objects.filter((obj) => obj.id !== id));
      setMarkers(markers.filter((marker) => marker.objId !== id));
      setPromptsByObject((prev) => {
        const updatedPrompts = { ...prev };
        delete updatedPrompts[id];
        return updatedPrompts;
      });
      if (selectedObjectId === id && objects.length > 1) {
        setSelectedObjectId(objects[0].id);
      } else if (objects.length === 1 && selectedObjectId === id) {
        setSelectedObjectId(null);
      }
      setFrameVersion((prev) => prev + 1);
      alert(`Object ${id} has been deleted.`);
    } catch (error) {
      console.error('Error deleting object:', error);
      alert('Error deleting object.');
    }
  };

  // Mute/Unmute Object
  const toggleMuteObject = async (id) => {
    try {
      const objIndex = objects.findIndex((obj) => obj.id === id);
      const newMutedState = !objects[objIndex].muted;
      await axios.post('/mute_object', {
        video_id: videoId,
        obj_id: id,
        muted: newMutedState,
      });
      setObjects((prevObjects) =>
        prevObjects.map((obj) =>
          obj.id === id ? { ...obj, muted: newMutedState } : obj
        )
      );
      fetchMasks(currentFrameIndex);
    } catch (error) {
      console.error('Error toggling mute state:', error);
      alert('Error toggling mute state.');
    }
  };

  // Handle Mask Opacity Change
  const handleMaskOpacityChange = (event, value) => {
    setObjects(
      objects.map((obj) =>
        obj.id === selectedObjectId ? { ...obj, maskOpacity: value } : obj
      )
    );
    fetchMasks(currentFrameIndex);
  };

  // Rename Object
  const renameObject = (id, newName) => {
    setObjects(
      objects.map((obj) => (obj.id === id ? { ...obj, name: newName } : obj))
    );
  };

  const moveEffect = (dragIndex, hoverIndex) => {
    const selectedObject = objects.find((obj) => obj.id === selectedObjectId);
    const updatedEffects = [...selectedObject.effects];
    const [removed] = updatedEffects.splice(dragIndex, 1);
    updatedEffects.splice(hoverIndex, 0, removed);
    setObjects((prevObjects) =>
      prevObjects.map((obj) =>
        obj.id === selectedObjectId ? { ...obj, effects: updatedEffects } : obj
      )
    );
    // Apply effects after reordering
    if (effectsMode) {
      applyEffects();
    }
  };

  const deleteEffect = (effectIndex) => {
    setObjects((prevObjects) =>
      prevObjects.map((obj) =>
        obj.id === selectedObjectId
          ? {
              ...obj,
              effects: obj.effects.filter((_, idx) => idx !== effectIndex),
            }
          : obj
      )
    );
    // Apply effects after deleting effect
    if (effectsMode) {
      applyEffects();
    }
  };

  const toggleMuteEffect = (effectIndex) => {
    setObjects((prevObjects) =>
      prevObjects.map((obj) => {
        if (obj.id === selectedObjectId) {
          const updatedEffects = obj.effects.map((effect, idx) => {
            if (idx === effectIndex) {
              return {
                ...effect,
                muted: !effect.muted,
              };
            }
            return effect;
          });
          return { ...obj, effects: updatedEffects };
        }
        return obj;
      })
    );
    // Apply effects after muting/unmuting
    if (effectsMode) {
      applyEffects();
    }
  };


  const handleEffectParameterChange = (effectIndex, param, value) => {
    const selectedObject = objects.find((obj) => obj.id === selectedObjectId);
    if (!selectedObject) return;
  
    const effect = selectedObject.effects[effectIndex];
    if (!effect) return;
  
    // Get parameter configuration and validate
    const effectConfig = effectLibrary.find((e) => e.name === effect.name);
    const paramConfig = effectConfig?.defaultParams[param] || defaultParamConfigs[param];
    if (!paramConfig) return;
  
    // Parse and validate value
    const paramType = paramConfig.type;
    let parsedValue = paramType === 'int' ? parseInt(value) : parseFloat(value);
    
    // Clamp value to valid range
    if (paramConfig.min !== undefined) {
      parsedValue = Math.max(paramConfig.min, parsedValue);
    }
    if (paramConfig.max !== undefined) {
      parsedValue = Math.min(paramConfig.max, parsedValue);
    }
  
    // Update object state
    setObjects(prevObjects => 
      prevObjects.map(obj => {
        if (obj.id === selectedObjectId) {
          const updatedEffects = obj.effects.map((eff, idx) => {
            if (idx === effectIndex) {
              return {
                ...eff,
                params: {
                  ...eff.params,
                  [param]: parsedValue,
                },
              };
            }
            return eff;
          });
          return { ...obj, effects: updatedEffects };
        }
        return obj;
      })
    );
  };

  // Handle Image Load for Correct Dimensions
  const handleImageLoad = (e) => {
    const { naturalWidth, naturalHeight } = e.target;
    setImageDimensions({ width: naturalWidth, height: naturalHeight });
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.width = naturalWidth;
      canvas.height = naturalHeight;
    }
    fetchMasks(currentFrameIndex);
  };

  // When changing frames, reapply effects if in effects mode
  useEffect(() => {
    if (effectsMode) {
      applyEffects();
    }
  }, [currentFrameIndex, effectsMode, applyEffects]);

  // Add this useEffect after your existing useEffect hooks
// In VideoEditor.js

// In VideoEditor.js

useEffect(() => {
  const selectedObject = objects.find((obj) => obj.id === selectedObjectId);
  if (selectedObject && selectedObject.featherParams) {
    setFeatherParams(selectedObject.featherParams);
  }
}, [selectedObjectId, objects]);



  // Return statement
  return (
    <DndProvider backend={HTML5Backend}>
      <div
        className="flex flex-row min-h-screen p-4"
        style={{ backgroundColor: 'var(--mainBg)', color: 'var(--mainText)' }}
      >
        {/* Sidebar */}
        <div
          className="w-1/4 p-4 shadow-md overflow-y-auto"
          style={{ backgroundColor: 'var(--sidebarBg)' }}
        >
          {!effectsMode ? (
            <>
              {/* Objects Panel */}
              <h2 className="text-xl font-semibold mb-2">Objects</h2>
              <div>
                {objects.map((obj) => (
                  <div
                    key={obj.id}
                    className={`flex items-center justify-between p-2 mb-2 cursor-pointer ${
                      selectedObjectId === obj.id ? 'bg-blue-100' : ''
                    }`}
                    onClick={() => setSelectedObjectId(obj.id)}
                  >
                    <input
                      type="text"
                      value={obj.name}
                      onChange={(e) => renameObject(obj.id, e.target.value)}
                      className="border rounded px-2 py-1"
                    />
                    <div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleMuteObject(obj.id);
                        }}
                        className={`mr-2 ${
                          obj.muted ? 'text-gray-400' : 'text-blue-500'
                        }`}
                      >
                        {obj.muted ? 'Unmute' : 'Mute'}
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteObject(obj.id);
                        }}
                        className="text-red-500 hover:text-red-700"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                ))}
                <button
                  onClick={addObject}
                  className="w-full py-2 px-4 rounded mt-2"
                  style={{
                    backgroundColor: 'var(--primaryButtonBg)',
                    color: 'var(--primaryButtonText)',
                  }}
                  onMouseOver={(e) =>
                    (e.currentTarget.style.backgroundColor = 'var(--primaryButtonHoverBg)')
                  }
                  onMouseOut={(e) =>
                    (e.currentTarget.style.backgroundColor = 'var(--primaryButtonBg)')
                  }
                >
                  Add New Object
                </button>
              </div>

              {/* Prompts Panel */}
              <div className="mt-4">
                <div className="flex space-x-2 mb-2">
                  <button
                    onClick={() => setPromptMode('point')}
                    className="flex-1 py-2 px-4 rounded"
                    style={{
                      backgroundColor:
                        promptMode === 'point' ? 'var(--buttonBgSelected)' : 'var(--buttonBgUnselected)',
                      color: promptMode === 'point' ? 'var(--buttonTextSelected)' : 'var(--buttonTextUnselected)',
                    }}
                    onMouseOver={(e) => {
                      if (promptMode !== 'point') {
                        e.currentTarget.style.backgroundColor = 'var(--buttonBgHover)';
                      }
                    }}
                    onMouseOut={(e) => {
                      if (promptMode !== 'point') {
                        e.currentTarget.style.backgroundColor = 'var(--buttonBgUnselected)';
                      }
                    }}
                  >
                    Point
                  </button>
                  <button
                    onClick={() => setPromptMode('box')}
                    className="flex-1 py-2 px-4 rounded"
                    style={{
                      backgroundColor:
                        promptMode === 'box' ? 'var(--buttonBgSelected)' : 'var(--buttonBgUnselected)',
                      color: promptMode === 'box' ? 'var(--buttonTextSelected)' : 'var(--buttonTextUnselected)',
                    }}
                    onMouseOver={(e) => {
                      if (promptMode !== 'box') {
                        e.currentTarget.style.backgroundColor = 'var(--buttonBgHover)';
                      }
                    }}
                    onMouseOut={(e) => {
                      if (promptMode !== 'box') {
                        e.currentTarget.style.backgroundColor = 'var(--buttonBgUnselected)';
                      }
                    }}
                  >
                    Box
                  </button>
                </div>

                <div className="flex space-x-2">
                  <button
                    onClick={() => setPromptType('positive')}
                    className="flex-1 py-2 px-4 rounded"
                    style={{
                      backgroundColor:
                        promptType === 'positive' ? 'var(--buttonBgSelected)' : 'var(--buttonBgUnselected)',
                      color: promptType === 'positive' ? 'var(--buttonTextSelected)' : 'var(--buttonTextUnselected)',
                    }}
                    onMouseOver={(e) => {
                      if (promptType !== 'positive') {
                        e.currentTarget.style.backgroundColor = 'var(--buttonBgHover)';
                      }
                    }}
                    onMouseOut={(e) => {
                      if (promptType !== 'positive') {
                        e.currentTarget.style.backgroundColor = 'var(--buttonBgUnselected)';
                      }
                    }}
                  >
                    Positive (+)
                  </button>
                  <button
                    onClick={() => setPromptType('negative')}
                    className="flex-1 py-2 px-4 rounded"
                    style={{
                      backgroundColor:
                        promptType === 'negative' ? 'var(--buttonBgSelected)' : 'var(--buttonBgUnselected)',
                      color: promptType === 'negative' ? 'var(--buttonTextSelected)' : 'var(--buttonTextUnselected)',
                    }}
                    onMouseOver={(e) => {
                      if (promptType !== 'negative') {
                        e.currentTarget.style.backgroundColor = 'var(--buttonBgHover)';
                      }
                    }}
                    onMouseOut={(e) => {
                      if (promptType !== 'negative') {
                        e.currentTarget.style.backgroundColor = 'var(--buttonBgUnselected)';
                      }
                    }}
                  >
                    Negative (-)
                  </button>
                </div>
              </div>
              <button
                onClick={trackObjects}
                className="w-full py-2 px-4 rounded mt-4"
                style={{
                  backgroundColor: trackingInProgress
                    ? 'var(--disabledButtonBg)'
                    : 'var(--primaryButtonBg)',
                  color: trackingInProgress
                    ? 'var(--disabledButtonText)'
                    : 'var(--primaryButtonText)',
                  cursor: trackingInProgress ? 'not-allowed' : 'pointer',
                }}
                disabled={trackingInProgress}
                onMouseOver={(e) => {
                  if (!trackingInProgress) {
                    e.currentTarget.style.backgroundColor = 'var(--primaryButtonHoverBg)';
                  }
                }}
                onMouseOut={(e) => {
                  if (!trackingInProgress) {
                    e.currentTarget.style.backgroundColor = 'var(--primaryButtonBg)';
                  }
                }}
              >
                {trackingInProgress ? 'Tracking...' : 'Track Objects'}
              </button>
              <button
                onClick={toggleEffectsMode}
                className="w-full py-2 px-4 rounded mt-2"
                style={{
                  backgroundColor: 'var(--primaryButtonBg)',
                  color: 'var(--primaryButtonText)',
                }}
                onMouseOver={(e) =>
                  (e.currentTarget.style.backgroundColor = 'var(--primaryButtonHoverBg)')
                }
                onMouseOut={(e) =>
                  (e.currentTarget.style.backgroundColor = 'var(--primaryButtonBg)')
                }
              >
                Add Effects
              </button>
              <button
                onClick={sendBatchPrompts}
                className="w-full py-2 px-4 rounded mt-2"
                style={{
                  backgroundColor:
                    Object.keys(promptsByObject).length === 0
                      ? 'var(--disabledButtonBg)'
                      : 'var(--primaryButtonBg)',
                  color:
                    Object.keys(promptsByObject).length === 0
                      ? 'var(--disabledButtonText)'
                      : 'var(--primaryButtonText)',
                  cursor:
                    Object.keys(promptsByObject).length === 0 ? 'not-allowed' : 'pointer',
                }}
                disabled={Object.keys(promptsByObject).length === 0}
                onMouseOver={(e) => {
                  if (Object.keys(promptsByObject).length !== 0) {
                    e.currentTarget.style.backgroundColor = 'var(--primaryButtonHoverBg)';
                  }
                }}
                onMouseOut={(e) => {
                  if (Object.keys(promptsByObject).length !== 0) {
                    e.currentTarget.style.backgroundColor = 'var(--primaryButtonBg)';
                  }
                }}
              >
                Apply Prompts
              </button>
            </>
          ) : (
            <>
              {/* Effects Panel */}
              <h2 className="text-xl font-semibold mb-2">Effects</h2>

              {/* Load Preset Modal */}
              <Modal
                open={isLoadPresetOpen}
                onClose={() => setIsLoadPresetOpen(false)}
                aria-labelledby="load-preset-modal"
                aria-describedby="modal-to-load-preset"
              >
                <Box
                  sx={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    width: 400,
                    bgcolor: 'background.paper',
                    border: '2px solid #000',
                    boxShadow: 24,
                    p: 4,
                  }}
                >
                  <h2 id="load-preset-modal" className="text-xl font-semibold mb-4">
                    Load Effect Stack Preset
                  </h2>
                  <div className="mb-4 max-h-64 overflow-y-auto">
                    {effectStackPresets.map((preset) => (
                      <div
                        key={preset.preset_name}
                        onClick={() => {
                          loadEffectStack(preset.effects_stack);
                          setIsLoadPresetOpen(false);
                        }}
                        className="cursor-pointer px-2 py-1 hover:bg-gray-200"
                      >
                        {preset.sub_folder && <span>[{preset.sub_folder}] </span>}
                        {preset.preset_name}
                      </div>
                    ))}
                  </div>
                  <div className="flex justify-end">
                    <button
                      onClick={() => setIsLoadPresetOpen(false)}
                      className="bg-gray-300 text-black py-2 px-4 rounded hover:bg-gray-400"
                    >
                      Close
                    </button>
                  </div>
                </Box>
              </Modal>

              {/* Continue with the rest of the effects panel */}
              <div className="mb-4">
                <h3 className="text-lg font-semibold mb-2">Select Object</h3>
                <div className="flex flex-wrap space-x-2">
                  {objects.map((obj) => (
                    <button
                      key={obj.id}
                      onClick={() => setSelectedObjectId(obj.id)}
                      className="py-2 px-4 rounded"
                      style={{
                        backgroundColor:
                          selectedObjectId === obj.id ? 'var(--primaryButtonBg)' : '#e5e7eb',
                        color:
                          selectedObjectId === obj.id ? 'var(--primaryButtonText)' : '#1f2937',
                      }}
                      onMouseOver={(e) => {
                        if (selectedObjectId !== obj.id) {
                          e.currentTarget.style.backgroundColor = '#d1d5db';
                        }
                      }}
                      onMouseOut={(e) => {
                        if (selectedObjectId !== obj.id) {
                          e.currentTarget.style.backgroundColor = '#e5e7eb';
                        }
                      }}
                    >
                      {obj.name}
                    </button>
                  ))}
                </div>
              </div>
              <div>
                <h3 className="text-lg font-semibold mb-2">Mask Opacity</h3>
                <Slider
                  value={
                    objects.find((obj) => obj.id === selectedObjectId)?.maskOpacity || 0.5
                  }
                  onChange={handleMaskOpacityChange}
                  min={0}
                  max={1}
                  step={0.01}
                  sx={{ color: 'var(--sliderColor)' }}
                />
              </div>

              {/* Feathering Controls */}
              <div className="mt-4 border border-gray-300 p-4 rounded-md">
                <h3 className="text-lg font-semibold mb-2">Feathering Controls</h3>
                {['radius', 'expand', 'opacity'].map((param) => (
                  <div key={param} className="mt-2">
                    <label className="block text-sm font-medium">
                      {param.charAt(0).toUpperCase() + param.slice(1)}: {featherParams[param]?.toFixed(2)}
                    </label>
                    <div className="flex items-center">
                      <Slider
                        value={featherParams[param]}
                        onChange={(e, value) => handleSliderChange(param, value)}
                        min={param === 'expand' ? -20 : param === 'opacity' ? 0 : 0}
                        max={param === 'expand' ? 20 : param === 'opacity' ? 1 : 100}
                        step={param === 'opacity' ? 0.1 : 1}
                        className="mr-2 flex-1"
                        sx={{ color: 'var(--sliderColor)' }} // Dynamic color using CSS variable
                      />
                      {/* Number input for direct typing */}
                      <input
                        type="number"
                        value={featherParams[param]}
                        onChange={(e) => handleSliderChange(param, e.target.value)}
                        min={param === 'expand' ? -20 : param === 'opacity' ? 0 : 0}
                        max={param === 'expand' ? 20 : param === 'opacity' ? 1 : 100}
                        step={param === 'opacity' ? 0.1 : 1}
                        className="w-16 ml-2 border border-gray-300 rounded px-2 py-1"
                      />
                    </div>
                  </div>
                ))}

                {/* Invert Mask Controls */}
                <div className="mt-4 border-t border-gray-300 pt-4">
                  <h3 className="text-lg font-semibold mb-2">Invert Mask Controls</h3>
                  <div className="mb-4">
                    <label className="inline-flex items-center">
                      <input
                        type="checkbox"
                        checked={featherParams.invert_mask}
                        onChange={(e) => handleCheckboxChange('invert_mask', e.target.checked)}
                        className="form-checkbox"
                      />
                      <span className="ml-2">Invert Mask</span>
                    </label>
                  </div>
                  <div className="mb-4">
                    <label className="block text-sm font-medium mb-1">Invert Intensity</label>
                    <div className="flex items-center">
                      <Slider
                        value={featherParams.invert_intensity}
                        onChange={(e, value) => handleSliderChange('invert_intensity', value)}
                        min={0}
                        max={1}
                        step={0.1}
                        className="mr-2 flex-1"
                        sx={{ color: 'var(--sliderColor)' }} // Dynamic color using CSS variable
                      />
                      {/* Number input for direct typing */}
                      <input
                        type="number"
                        value={featherParams.invert_intensity}
                        onChange={(e) => handleSliderChange('invert_intensity', e.target.value)}
                        min={0}
                        max={1}
                        step={0.1}
                        className="w-16 ml-2 border border-gray-300 rounded px-2 py-1"
                      />
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-4 mb-4">
                <h3 className="text-lg font-semibold mb-2">Preview Mode</h3>
                <div className="flex space-x-2">
                  <button
                    onClick={() => setEffectPreviewMode('all')}
                    className="flex-1 py-2 px-4 rounded"
                    style={{
                      backgroundColor: effectPreviewMode === 'all' ? 'var(--buttonBgSelected)' : 'var(--buttonBgUnselected)',
                      color: effectPreviewMode === 'all' ? 'var(--buttonTextSelected)' : 'var(--buttonTextUnselected)',
                    }}
                    onMouseOver={(e) => {
                      if (effectPreviewMode !== 'all') {
                        e.currentTarget.style.backgroundColor = 'var(--buttonBgHover)';
                      }
                    }}
                    onMouseOut={(e) => {
                      if (effectPreviewMode !== 'all') {
                        e.currentTarget.style.backgroundColor = 'var(--buttonBgUnselected)';
                      }
                    }}
                  >
                    Show All
                  </button>
                  <button
                    onClick={() => setEffectPreviewMode('selected')}
                    className="flex-1 py-2 px-4 rounded"
                    style={{
                      backgroundColor: effectPreviewMode === 'selected' ? 'var(--buttonBgSelected)' : 'var(--buttonBgUnselected)',
                      color: effectPreviewMode === 'selected' ? 'var(--buttonTextSelected)' : 'var(--buttonTextUnselected)',
                    }}
                    onMouseOver={(e) => {
                      if (effectPreviewMode !== 'selected') {
                        e.currentTarget.style.backgroundColor = 'var(--buttonBgHover)';
                      }
                    }}
                    onMouseOut={(e) => {
                      if (effectPreviewMode !== 'selected') {
                        e.currentTarget.style.backgroundColor = 'var(--buttonBgUnselected)';
                      }
                    }}
                  >
                    Selected Only
                  </button>
                </div>
              </div>


              {/* Effect Stack and Buttons */}
              <div className="mt-4">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-lg font-semibold">Effect Stack</h3>
                  {/* Buttons Next to Effect Stack */}
                  <div className="flex items-center">
                    <button
                      onClick={saveEffectStackPreset}
                      className="px-2 py-1 rounded mr-1 text-sm"
                      style={{
                        backgroundColor: 'var(--primaryButtonBg)',
                        color: 'var(--primaryButtonText)',
                        height: '24px',
                      }}
                      onMouseOver={(e) =>
                        (e.currentTarget.style.backgroundColor = 'var(--primaryButtonHoverBg)')
                      }
                      onMouseOut={(e) =>
                        (e.currentTarget.style.backgroundColor = 'var(--primaryButtonBg)')
                      }
                    >
                      Save Preset
                    </button>
                    <button
                      onClick={() => setIsChainMode(!isChainMode)}
                      className="px-2 py-1 rounded mr-1 text-sm"
                      style={{
                        backgroundColor: isChainMode ? '#10b981' : '#e5e7eb',
                        color: isChainMode ? '#ffffff' : '#1f2937',
                        height: '24px',
                      }}
                      onMouseOver={(e) => {
                        if (!isChainMode) {
                          e.currentTarget.style.backgroundColor = '#d1d5db';
                        }
                      }}
                      onMouseOut={(e) => {
                        if (!isChainMode) {
                          e.currentTarget.style.backgroundColor = '#e5e7eb';
                        }
                      }}
                    >
                      <FontAwesomeIcon icon={faLink} />
                    </button>
                    <button
                      onClick={() => setIsLoadPresetOpen(true)}
                      className="px-2 py-1 rounded text-sm"
                      style={{
                        backgroundColor: 'var(--primaryButtonBg)',
                        color: 'var(--primaryButtonText)',
                        height: '24px',
                      }}
                      onMouseOver={(e) =>
                        (e.currentTarget.style.backgroundColor = 'var(--primaryButtonHoverBg)')
                      }
                      onMouseOut={(e) =>
                        (e.currentTarget.style.backgroundColor = 'var(--primaryButtonBg)')
                      }
                    >
                      Load Preset
                    </button>
                  </div>
                </div>
                {/* Render Effects */}
                {/* Effects Rack */}
                {objects.find((obj) => obj.id === selectedObjectId)?.effects.map((effect, index) => (
                  <EffectBlock
                    key={index}
                    effect={effect}
                    index={index}
                    moveEffect={moveEffect}
                    deleteEffect={deleteEffect}
                    toggleMuteEffect={toggleMuteEffect}
                    handleEffectParameterChange={handleEffectParameterChange}
                    handleEffectPropertyChange={handleEffectPropertyChange}
                    effectLibrary={effectLibrary}
                    defaultParamConfigs={defaultParamConfigs}
                    colorPresets={colorPresets}
                    saveColorPreset={saveColorPreset}
                  />
                ))}
                <select
                  value={selectedEffectName}
                  onChange={(e) => setSelectedEffectName(e.target.value)}
                  className="w-full py-2 px-4 rounded mt-2"
                  style={{
                    backgroundColor: '#0f1920',
                    borderColor: '#096592',
                    color: '#128eb8',
                    borderWidth: '1px',
                    borderStyle: 'solid',
                  }}
                >
                  <option value="" disabled style={{ color: '#128eb8' }}>
                    {isLoadingEffects ? 'Loading Effects...' : 'Select Effect'}
                  </option>
                  {effectLibrary.map((effect) => (
                    <option key={effect.name} value={effect.name} style={{ color: '#128eb8' }}>
                      {effect.name}
                    </option>
                  ))}
                </select>

                <button
                  onClick={addEffect}
                  className="w-full py-2 px-4 rounded mt-2"
                  style={{
                    backgroundColor: 'var(--primaryButtonBg)',
                    color: 'var(--primaryButtonText)',
                  }}
                  onMouseOver={(e) =>
                    (e.currentTarget.style.backgroundColor = 'var(--primaryButtonHoverBg)')
                  }
                  onMouseOut={(e) =>
                    (e.currentTarget.style.backgroundColor = 'var(--primaryButtonBg)')
                  }
                >
                  Load Effect
                </button>

                {/* Import Effect Button */}
                <button
                  onClick={handleOpenImportModal}
                  className="w-full py-2 px-4 rounded mt-2"
                  style={{
                    backgroundColor: 'var(--primaryButtonBg)',
                    color: 'var(--primaryButtonText)',
                  }}
                  onMouseOver={(e) =>
                    (e.currentTarget.style.backgroundColor = 'var(--primaryButtonHoverBg)')
                  }
                  onMouseOut={(e) =>
                    (e.currentTarget.style.backgroundColor = 'var(--primaryButtonBg)')
                  }
                >
                  Import Effect
                </button>

                {/* Import Effect Modal */}
                <Modal
                  open={importModalOpen}
                  onClose={handleCloseImportModal}
                  aria-labelledby="import-effect-modal"
                  aria-describedby="modal-to-import-new-effect"
                >
                  <Box
                    sx={{
                      position: 'absolute',
                      top: '50%',
                      left: '50%',
                      transform: 'translate(-50%, -50%)',
                      width: 400,
                      bgcolor: 'background.paper',
                      border: '2px solid #000',
                      boxShadow: 24,
                      p: 4,
                    }}
                  >
                    <h2 id="import-effect-modal" className="text-xl font-semibold mb-4">
                      Import New Effect
                    </h2>
                    <div className="mb-4">
                      <label className="block text-sm font-medium mb-1">Effect Name</label>
                      <input
                        type="text"
                        value={newEffectName}
                        onChange={(e) => setNewEffectName(e.target.value)}
                        className="w-full border rounded px-2 py-1"
                        placeholder="Enter effect name"
                      />
                    </div>
                    <div className="mb-4">
                      <label className="block text-sm font-medium mb-1">Effect Code</label>
                      <textarea
                        value={newEffectCode}
                        onChange={(e) => setNewEffectCode(e.target.value)}
                        className="w-full border rounded px-2 py-1"
                        rows={10}
                        placeholder="Paste your OpenFX code here"
                      ></textarea>
                    </div>
                    <div className="mb-4">
                      <label className="block text-sm font-medium mb-1">
                        Effect Config (Optional, JSON)
                      </label>
                      <textarea
                        value={newEffectConfig}
                        onChange={(e) => setNewEffectConfig(e.target.value)}
                        className="w-full border rounded px-2 py-1"
                        rows={5}
                        placeholder='Paste your effect config in JSON format, e.g., { "param1": "value1" }'
                      ></textarea>
                    </div>
                    <div className="flex justify-end">
                      <button
                        onClick={handleCloseImportModal}
                        className="bg-gray-300 text-black py-2 px-4 rounded hover:bg-gray-400 mr-2"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={handleImportEffect}
                        className="bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600"
                      >
                        Import
                      </button>
                    </div>
                  </Box>
                </Modal>
              </div>

              <div className="flex space-x-2 mt-4">
                <button
                  onClick={toggleEffectsMode}
                  className="flex-1 py-2 px-4 rounded"
                  style={{
                    backgroundColor: 'var(--primaryButtonBg)',
                    color: 'var(--primaryButtonText)',
                  }}
                  onMouseOver={(e) =>
                    (e.currentTarget.style.backgroundColor = 'var(--primaryButtonHoverBg)')
                  }
                  onMouseOut={(e) =>
                    (e.currentTarget.style.backgroundColor = 'var(--primaryButtonBg)')
                  }
                >
                  Back to Objects
                </button>
                <button
                  onClick={() => exportVideo({ video_with_effects: true })}
                  className="flex-1 py-2 px-4 rounded"
                  style={{
                    backgroundColor: 'var(--primaryButtonBg)',
                    color: 'var(--primaryButtonText)',
                  }}
                  onMouseOver={(e) =>
                    (e.currentTarget.style.backgroundColor = 'var(--primaryButtonHoverBg)')
                  }
                  onMouseOut={(e) =>
                    (e.currentTarget.style.backgroundColor = 'var(--primaryButtonBg)')
                  }
                >
                  Export Video
                </button>
                <button
                  onClick={() => exportVideo({ masks: true })}
                  className="flex-1 py-2 px-4 rounded"
                  style={{
                    backgroundColor: 'var(--primaryButtonBg)',
                    color: 'var(--primaryButtonText)',
                  }}
                  onMouseOver={(e) =>
                    (e.currentTarget.style.backgroundColor = 'var(--primaryButtonHoverBg)')
                  }
                  onMouseOut={(e) =>
                    (e.currentTarget.style.backgroundColor = 'var(--primaryButtonBg)')
                  }
                >
                  Export Masks
                </button>
              </div>
              {/* Render All Frames Button */}
              <button
                onClick={applyEffectsToAllFrames}
                className="w-full py-2 px-4 rounded mt-2"
                style={{
                  backgroundColor: 'var(--primaryButtonBg)',
                  color: 'var(--primaryButtonText)',
                }}
                onMouseOver={(e) =>
                  (e.currentTarget.style.backgroundColor = 'var(--primaryButtonHoverBg)')
                }
                onMouseOut={(e) =>
                  (e.currentTarget.style.backgroundColor = 'var(--primaryButtonBg)')
                }
              >
                Render All Frames
              </button>
            </>
          )}
          <button
            onClick={resetState}
            className="w-full py-2 px-4 rounded mt-2"
            style={{
              backgroundColor: 'var(--primaryButtonBg)',
              color: 'var(--primaryButtonText)',
            }}
            onMouseOver={(e) =>
              (e.currentTarget.style.backgroundColor = 'var(--primaryButtonHoverBg)')
            }
            onMouseOut={(e) =>
              (e.currentTarget.style.backgroundColor = 'var(--primaryButtonBg)')
            }
          >
            Reset All
          </button>

          {/* Color Settings Button */}
          <button
            onClick={() => setIsGlobalColorModalOpen(true)}
            className="w-full py-2 px-4 rounded mt-2"
            style={{
              backgroundColor: 'var(--primaryButtonBg)',
              color: 'var(--primaryButtonText)',
            }}
            onMouseOver={(e) =>
              (e.currentTarget.style.backgroundColor = 'var(--primaryButtonHoverBg)')
            }
            onMouseOut={(e) =>
              (e.currentTarget.style.backgroundColor = 'var(--primaryButtonBg)')
            }
          >
            Color Settings
          </button>

          {/* Global Color Settings Modal */}
          <Modal
            open={isGlobalColorModalOpen}
            onClose={() => setIsGlobalColorModalOpen(false)}
            aria-labelledby="global-color-settings-modal"
            aria-describedby="modal-for-global-color-settings"
          >
            <Box
              sx={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                width: 600,
                bgcolor: 'background.paper',
                border: '2px solid #000',
                boxShadow: 24,
                p: 4,
                maxHeight: '80vh',
                overflowY: 'auto',
              }}
            >
              <h2 id="global-color-settings-modal" className="text-xl font-semibold mb-4">
                Global Color Settings
              </h2>
              {/* Render Categories */}
              {colorCategories.map((category) => (
                <div key={category.categoryName} className="mb-4">
                  <h3 className="text-lg font-semibold mb-2">{category.categoryName}</h3>
                  {category.components.map((component) => (
                    <div key={component.key} className="flex items-center mb-2">
                      <input
                        type="checkbox"
                        checked={!!globalColorSettings[component.key]}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setGlobalColorSettings((prev) => ({
                              ...prev,
                              [component.key]: '#ffffff', // Default color
                            }));
                          } else {
                            setGlobalColorSettings((prev) => {
                              const newSettings = { ...prev };
                              delete newSettings[component.key];
                              return newSettings;
                            });
                          }
                        }}
                        className="mr-2"
                      />
                      <span className="flex-1">{component.name}</span>
                      {globalColorSettings[component.key] && (
                        <div className="ml-2">
                          <SketchPicker
                            color={globalColorSettings[component.key]}
                            onChangeComplete={(color) =>
                              setGlobalColorSettings((prev) => ({
                                ...prev,
                                [component.key]: color.hex,
                              }))
                            }
                          />
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ))}

              {/* Save and Load Preset Buttons */}
              <div className="flex items-center justify-between mt-4">
                <div>
                  <input
                    type="text"
                    value={newColorPresetName}
                    onChange={(e) => setNewColorPresetName(e.target.value)}
                    className="border rounded px-2 py-1 mr-2"
                    placeholder="Preset Name"
                  />
                  <button
                    onClick={saveColorSettingsPreset}
                    className="bg-green-500 text-white py-2 px-4 rounded hover:bg-green-600"
                  >
                    Save Preset
                  </button>
                </div>
                <button
                  onClick={() => setIsLoadColorPresetOpen(true)}
                  className="bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600"
                >
                  Load Preset
                </button>
              </div>

              {/* Load Color Preset Modal */}
              <Modal
                open={isLoadColorPresetOpen}
                onClose={() => setIsLoadColorPresetOpen(false)}
                aria-labelledby="load-color-preset-modal"
                aria-describedby="modal-to-load-color-preset"
              >
                <Box
                  sx={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    width: 400,
                    bgcolor: 'background.paper',
                    border: '2px solid #000',
                    boxShadow: 24,
                    p: 4,
                  }}
                >
                  <h2 id="load-color-preset-modal" className="text-xl font-semibold mb-4">
                    Load Color Preset
                  </h2>
                  <div className="mb-4 max-h-64 overflow-y-auto">
                    {colorPresets.map((preset) => (
                      <div
                        key={preset.preset_name}
                        onClick={() => loadColorSettingsPreset(preset)}
                        className="cursor-pointer px-2 py-1 hover:bg-gray-200"
                      >
                        {preset.preset_name}
                      </div>
                    ))}
                  </div>
                  <div className="flex justify-end">
                    <button
                      onClick={() => setIsLoadColorPresetOpen(false)}
                      className="bg-gray-300 text-black py-2 px-4 rounded hover:bg-gray-400"
                    >
                      Close
                    </button>
                  </div>
                </Box>
              </Modal>

              <div className="flex justify-end mt-4">
                <button
                  onClick={() => setIsGlobalColorModalOpen(false)}
                  className="bg-gray-300 text-black py-2 px-4 rounded hover:bg-gray-400 mr-2"
                >
                  Cancel
                </button>
                <button
                  onClick={() => {
                    applyGlobalColors(globalColorSettings);
                    setIsGlobalColorModalOpen(false);
                  }}
                  className="bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600"
                >
                  Apply
                </button>
              </div>
            </Box>
          </Modal>
        </div>

            {/* Main Video Display */}
            <div className="flex flex-col items-center w-3/4">
              <h1 className="text-2xl font-bold mb-4">Video Editor</h1>
              <div className="relative">
                {frames.length > 0 && (
                  <>
                    <img
                      key={`${currentFrameIndex}-${frameVersion}`}
                      src={`${frames[currentFrameIndex]}?v=${frameVersion}`}
                      alt="Current Frame"
                      onClick={handleFrameClick}
                      onLoad={handleImageLoad}
                      className="border border-gray-300 cursor-crosshair"
                      style={{ maxWidth: '100%', height: 'auto' }}
                    />
                    <canvas
                      ref={canvasRef}
                      className="absolute top-0 left-0"
                      style={{
                        pointerEvents: 'none',
                        width: '100%',
                        height: '100%',
                      }}
                    />
                    {concurrentEffectsEnabled && (
                      // In VideoEditor.jsx
                      <ConcurrentEffectsManager
                        objects={objects}
                        selectedObjectId={selectedObjectId}
                        effectPreviewMode={effectPreviewMode}
                        onEffectUpdate={() => setFrameVersion(prev => prev + 1)}  // This is the one you're actually using
                        videoId={videoId}
                        currentFrameIndex={currentFrameIndex}
                        effectsMode={effectsMode}
                      />
                    )}
                  </>
                )}
              </div>
              <div className="flex items-center justify-center mt-4 space-x-4">
            <button
              onClick={prevFrame}
              className="py-2 px-4 rounded"
              style={{
                backgroundColor: 'var(--primaryButtonBg)',
                color: 'var(--primaryButtonText)',
              }}
              onMouseOver={(e) =>
                (e.currentTarget.style.backgroundColor = 'var(--primaryButtonHoverBg)')
              }
              onMouseOut={(e) =>
                (e.currentTarget.style.backgroundColor = 'var(--primaryButtonBg)')
              }
            >
              Previous Frame
            </button>
            <button
              onClick={togglePlay}
              className="py-2 px-4 rounded"
              style={{
                backgroundColor: 'var(--primaryButtonBg)',
                color: 'var(--primaryButtonText)',
              }}
              onMouseOver={(e) =>
                (e.currentTarget.style.backgroundColor = 'var(--primaryButtonHoverBg)')
              }
              onMouseOut={(e) =>
                (e.currentTarget.style.backgroundColor = 'var(--primaryButtonBg)')
              }
            >
              {playing ? 'Pause' : 'Play'}
            </button>
            <button
              onClick={nextFrame}
              className="py-2 px-4 rounded"
              style={{
                backgroundColor: 'var(--primaryButtonBg)',
                color: 'var(--primaryButtonText)',
              }}
              onMouseOver={(e) =>
                (e.currentTarget.style.backgroundColor = 'var(--primaryButtonHoverBg)')
              }
              onMouseOut={(e) =>
                (e.currentTarget.style.backgroundColor = 'var(--primaryButtonBg)')
              }
            >
              Next Frame
            </button>
          </div>
          <div className="w-full mt-4">
            <input
              type="range"
              min="0"
              max={frames.length - 1}
              value={currentFrameIndex}
              onChange={(e) => setCurrentFrameIndex(Number(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between mt-2">
              <span>
                Frame: {currentFrameIndex + 1} / {frames.length}
              </span>
            </div>
          </div>
        </div>
      </div>
    </DndProvider>
  );
}

export default VideoEditor;

