// UploadVideo.jsx
import React, { useState } from 'react';
import axios from 'axios';

function UploadVideo({ setVideoId }) {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);

    setUploading(true);

    try {
      // Use a relative URL so that the request goes through your proxy setting in package.json.
      const response = await axios.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      // Set video ID from response data
      setVideoId(response.data.video_id);
    } catch (error) {
      console.error('Error uploading video:', error);
      alert('Error uploading video');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <h1 className="text-3xl font-bold mb-8">Upload a Video to Start Editing</h1>
      <form onSubmit={handleSubmit} className="flex flex-col items-center">
        <input
          type="file"
          accept="video/*"
          onChange={(e) => setFile(e.target.files[0])}
          className="file-input file-input-bordered w-full max-w-xs mb-4"
        />
        <button
          type="submit"
          className={`btn btn-primary ${uploading ? 'loading' : ''}`}
          disabled={!file || uploading}
        >
          {uploading ? 'Uploading...' : 'Upload Video'}
        </button>
      </form>
    </div>
  );
}

export default UploadVideo;
