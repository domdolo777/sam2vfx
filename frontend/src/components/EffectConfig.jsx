import React from 'react';

const EffectConfig = ({ params, setParams, defaultParams }) => {
  const handleChange = (paramName, value) => {
    setParams(paramName, value);
  };

  return (
    <div className="effect-config">
      <h4>Configure Effect</h4>
      {defaultParams && Object.entries(defaultParams).map(([paramName, paramDetails]) => (
        <div key={paramName} className="param-control">
          <label>{paramName}</label>
          <div className="flex items-center">
            <input
              type={paramDetails.type === 'int' ? 'range' : 'range'}
              min={paramDetails.min}
              max={paramDetails.max}
              step={paramDetails.type === 'int' ? '1' : '0.1'}
              value={params[paramName]}
              onChange={(e) => handleChange(paramName, paramDetails.type === 'int' ? parseInt(e.target.value) : parseFloat(e.target.value))}
              className="mr-2 flex-1"
            />
            <input
              type="number"
              min={paramDetails.min}
              max={paramDetails.max}
              step={paramDetails.type === 'int' ? '1' : '0.1'}
              value={params[paramName]}
              onChange={(e) => handleChange(paramName, paramDetails.type === 'int' ? parseInt(e.target.value) : parseFloat(e.target.value))}
              className="w-16 ml-2 border border-gray-300 rounded px-2 py-1"
            />
          </div>
          <span>{params[paramName]}</span>
        </div>
      ))}
    </div>
  );
};

export default EffectConfig;