import React, { useState, useEffect } from 'react';
import EffectConfig from './EffectConfig';
import axios from 'axios';


const ApplyEffect = ({ videoId, selectedObjectId, objects, currentFrameIndex, featherParams, queueEffect }) => {
    const [effects, setEffects] = useState([]);
    const [selectedEffect, setSelectedEffect] = useState(null);
    const [effectParams, setEffectParams] = useState({});
    const [invertMask, setInvertMask] = useState(false); // Add state for Invert Mask
    const [invertIntensity, setInvertIntensity] = useState(1.0); // Add Invert Intensity state
    const [message, setMessage] = useState('');
    const [loadingEffects, setLoadingEffects] = useState(true);

    useEffect(() => {
      const fetchEffects = async () => {
          setLoadingEffects(true);
          try {
              const response = await axios.get(`/get_effects/${videoId}`);
              if (response.data && response.data.effects){
                  setEffects(response.data.effects);
              }
              else {
                  console.error("Invalid effect data:", response.data);
              }
          } catch (error) {
              console.error("Error fetching effects:", error);
          } finally {
              setLoadingEffects(false);
          }
      };

      fetchEffects();
  }, [videoId]);

    const handleEffectChange = (e) => {
        const effectName = e.target.value;
        const effect = effects.find(eff => eff.name === effectName);
        setSelectedEffect(effect);

        const defaults = {};
        if (effect && effect.defaultParams) {
            Object.entries(effect.defaultParams).forEach(([key, param]) => {
                defaults[key] = param.value;
            });
        }

        setEffectParams(defaults);

    };


    const handleParamChange = (paramName, value) => {
        setEffectParams(prev => ({ ...prev, [paramName]: value }));
    };


    const handleApplyEffect = () => {
        if (!selectedEffect) {
            setMessage('Please select an effect.');
            return;
        }

        queueEffect({
            video_id: videoId,
            obj_id: selectedObjectId,
            effects: [{
                name: selectedEffect.name,
                params: effectParams,
                muted: false, // Correctly set muted
                blend_mode: "Normal", // Default blend mode
                radius: featherParams.radius, // Pass radius
                expand: featherParams.expand, //Pass expand
                opacity: featherParams.opacity, //Pass Opacity
                invert_mask: featherParams.invert_mask, //Pass invert mask
                invert_intensity: featherParams.invert_intensity //Pass invert intensity
            }],
            feather_params: featherParams,
            frame_idx: currentFrameIndex,
            apply_to_all_frames: false,

        });


    };

    return (
        <div className="apply-effect">
            <h3>Apply Effect</h3>

            {/* Conditionally render loading message or effect selection */}
            {loadingEffects ? (
                <p>Loading Effects...</p>
            ) : (
                <select onChange={handleEffectChange} value={selectedEffect?.name || ''}>
                    <option value="" disabled>Select Effect</option>
                    {effects.map(effect => (
                        <option key={effect.name} value={effect.name}>{effect.name}</option>
                    ))}
                </select>
            )}

            {/* Conditionally render EffectConfig */}
            {selectedEffect && (
                <EffectConfig
                    params={effectParams}
                    setParams={handleParamChange}
                    defaultParams={selectedEffect.defaultParams}
                />
            )}
            {/* Invert Mask controls */}
            <div className="invert-options">
                <label>
                    <input
                        type="checkbox"
                        checked={invertMask}
                        onChange={(e) => setInvertMask(e.target.checked)}
                    />
                    Invert Mask
                </label>
                {/* Invert Intensity Control (Only shown when Invert Mask is checked) */}
                {invertMask && (
                    <div>
                        <label>Invert Intensity</label>
                        <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.1"
                            value={invertIntensity}
                            onChange={(e) => setInvertIntensity(parseFloat(e.target.value))}
                        />
                        <span>{invertIntensity}</span>
                    </div>
                )}
            </div>
            <button onClick={handleApplyEffect}>Apply Effect</button>
            {message && <p>{message}</p>}
        </div>

    );
};


export default ApplyEffect;
