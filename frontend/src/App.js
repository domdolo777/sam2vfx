// src/App.js

import React, { useState } from 'react';
import UploadVideo from './components/UploadVideo';
import VideoEditor from './components/VideoEditor';

function App() {
  const [videoId, setVideoId] = useState(null);

  return (
    <div className="App">
      {!videoId ? (
        <UploadVideo setVideoId={setVideoId} />
      ) : (
        <VideoEditor videoId={videoId} />
      )}
    </div>
  );
}

export default App;
