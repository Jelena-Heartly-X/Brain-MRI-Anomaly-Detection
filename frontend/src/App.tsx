import React, { useState, useEffect } from 'react';
import SidebarLeft from './components/SidebarLeft';
import CenterStage from './components/CenterStage';
import SidebarRight from './components/SidebarRight';
import { UploadCloud, Brain } from 'lucide-react';

export interface SliceData {
  id: number;
  base: string;
  score: number;
  label: number;
  subject: number;
  has_gt: boolean;
  region: string;
}

export interface Manifest {
  total_slices: number;
  architecture: string;
  dataset_train: string;
  dataset_test: string;
  metrics: { roc_auc: number; f1: number; note?: string };
  slices: SliceData[];
}

function App() {
  const [hasUploaded, setHasUploaded] = useState(false);
  const [currentSliceIdx, setCurrentSliceIdx] = useState(0);
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [loading, setLoading] = useState(false);

  // Load manifest.json from public/ when user clicks upload
  useEffect(() => {
    if (!hasUploaded) return;
    setLoading(true);
    fetch('/manifest.json')
      .then(r => r.json())
      .then((data: Manifest) => {
        setManifest(data);
        setCurrentSliceIdx(0);
        setLoading(false);
      })
      .catch(() => {
        // Fallback: use simulated data if manifest not found
        setManifest(null);
        setLoading(false);
      });
  }, [hasUploaded]);

  if (!hasUploaded) {
    return (
      <div className="app-container">
        <div className="upload-overlay">
          <div
            className="drop-zone animate-slide-down"
            onClick={() => setHasUploaded(true)}
          >
            <UploadCloud size={48} color="var(--color-accent)" style={{ margin: '0 auto 1rem auto' }} />
            <h2 style={{ marginBottom: '0.5rem' }}>Drop NIfTI scan or select file</h2>
            <p style={{ color: 'var(--text-secondary)' }}>Conv-MAE Inference Pipeline will start automatically</p>
          </div>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="app-container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ textAlign: 'center' }}>
          <Brain size={48} color="var(--color-accent)" style={{ margin: '0 auto 1rem auto', animation: 'pulse 1.5s infinite' }} />
          <p style={{ color: 'var(--text-secondary)' }}>Loading manifest and brain slices...</p>
        </div>
      </div>
    );
  }

  const slices = manifest?.slices ?? [];
  const totalSlices = slices.length || 50;

  return (
    <div className="app-container animate-fade-in">
      <SidebarLeft
        currentSlice={currentSliceIdx}
        setSlice={setCurrentSliceIdx}
        slices={slices}
        totalSlices={totalSlices}
      />
      <CenterStage
        currentSlice={currentSliceIdx}
        setSlice={setCurrentSliceIdx}
        slices={slices}
        totalSlices={totalSlices}
      />
      <SidebarRight
        currentSlice={currentSliceIdx}
        slices={slices}
        manifest={manifest}
      />
    </div>
  );
}

export default App;
